/**
 * Newsvendor Problem Backend - Node.js API Server
 * 
 * Implements RESTful API for stochastic inventory optimization
 * Features: Critical fractile calculation, Monte Carlo simulation,
 * profit curve generation, and database persistence
 */

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/newsvendor';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('✓ MongoDB connected successfully'))
.catch(err => console.error('✗ MongoDB connection error:', err));

// Mongoose Schemas
const ScenarioSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  parameters: {
    price: Number,
    cost: Number,
    disposal: Number,
    salvage: { type: Number, default: 0 },
    demandMean: Number,
    demandStd: Number,
    distribution: String,
    poissonLambda: Number,
    leadTime: { type: Number, default: 0 }
  },
  results: {
    optimalQ: Number,
    criticalFractile: Number,
    expectedProfit: Number,
    underageCost: Number,
    overageCost: Number
  },
  simulation: {
    iterations: Number,
    avgProfit: Number,
    stdDev: Number,
    percentiles: {
      p5: Number,
      p25: Number,
      p50: Number,
      p75: Number,
      p95: Number
    }
  }
});

const Scenario = mongoose.model('Scenario', ScenarioSchema);

// Statistical Functions

/**
 * Normal Distribution CDF using approximation
 */
function normalCDF(x, mean, std) {
  const z = (x - mean) / std;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

/**
 * Normal Distribution PDF
 */
function normalPDF(x, mean, std) {
  return (1 / (std * Math.sqrt(2 * Math.PI))) * 
         Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
}

/**
 * Poisson Distribution PMF
 */
function poissonPMF(k, lambda) {
  if (k < 0) return 0;
  if (k > 500) return 0; // Prevent overflow
  
  let result = Math.exp(-lambda);
  for (let i = 1; i <= k; i++) {
    result *= lambda / i;
  }
  return result;
}

/**
 * Poisson Distribution CDF
 */
function poissonCDF(k, lambda) {
  let sum = 0;
  for (let i = 0; i <= k; i++) {
    sum += poissonPMF(i, lambda);
  }
  return sum;
}

/**
 * Inverse Normal CDF (quantile function)
 */
function inverseNormalCDF(p, mean = 0, std = 1) {
  if (p <= 0 || p >= 1) throw new Error('Probability must be between 0 and 1');
  
  // Rational approximation for central region
  const a = [
    -3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00
  ];
  
  const b = [
    -5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01
  ];
  
  const c = [
    -7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
    4.374664141464968e+00, 2.938163982698783e+00
  ];
  
  const d = [
    7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00
  ];
  
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  
  let q, r, x;
  
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  
  return mean + std * x;
}

/**
 * Calculate optimal order quantity using critical fractile
 */
function calculateOptimalQ(params) {
  const { price, cost, disposal, salvage = 0, distribution, demandMean, demandStd, poissonLambda } = params;
  
  const cu = price - cost; // Underage cost
  const co = cost + disposal - salvage; // Overage cost
  const criticalFractile = cu / (cu + co);
  
  let optimalQ = 0;
  
  if (distribution === 'normal') {
    optimalQ = Math.round(inverseNormalCDF(criticalFractile, demandMean, demandStd));
  } else if (distribution === 'poisson') {
    // Binary search for optimal Q
    let low = 0;
    let high = poissonLambda * 3;
    
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      const cdf = poissonCDF(mid, poissonLambda);
      
      if (cdf < criticalFractile) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    optimalQ = low;
  }
  
  return {
    optimalQ: Math.max(0, optimalQ),
    criticalFractile,
    underageCost: cu,
    overageCost: co
  };
}

/**
 * Calculate expected profit for given order quantity
 */
function calculateExpectedProfit(Q, params) {
  const { price, cost, disposal, salvage = 0, distribution, demandMean, demandStd, poissonLambda } = params;
  
  let expectedSales = 0;
  let expectedLeftover = 0;
  const maxDemand = distribution === 'normal' ? demandMean * 3 : poissonLambda * 3;
  
  if (distribution === 'normal') {
    for (let d = 0; d <= maxDemand; d++) {
      const prob = normalPDF(d, demandMean, demandStd);
      expectedSales += Math.min(d, Q) * prob;
      expectedLeftover += Math.max(0, Q - d) * prob;
    }
  } else if (distribution === 'poisson') {
    for (let d = 0; d <= maxDemand; d++) {
      const prob = poissonPMF(d, poissonLambda);
      expectedSales += Math.min(d, Q) * prob;
      expectedLeftover += Math.max(0, Q - d) * prob;
    }
  }
  
  const revenue = price * expectedSales;
  const productionCost = cost * Q;
  const disposalCost = disposal * expectedLeftover;
  const salvageRevenue = salvage * expectedLeftover;
  
  return revenue - productionCost - disposalCost + salvageRevenue;
}

/**
 * Generate demand sample based on distribution
 */
function generateDemand(params) {
  const { distribution, demandMean, demandStd, poissonLambda } = params;
  
  if (distribution === 'normal') {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.max(0, Math.round(demandMean + z * demandStd));
  } else if (distribution === 'poisson') {
    const L = Math.exp(-poissonLambda);
    let k = 0;
    let p = 1;
    
    do {
      k++;
      p *= Math.random();
    } while (p > L && k < 1000);
    
    return k - 1;
  }
  
  return demandMean;
}

/**
 * Run Monte Carlo simulation
 */
function runMonteCarloSimulation(Q, params, iterations = 10000) {
  const { price, cost, disposal, salvage = 0 } = params;
  const profits = [];
  const scenarios = [];
  
  for (let i = 0; i < iterations; i++) {
    const demand = generateDemand(params);
    const sales = Math.min(demand, Q);
    const leftover = Math.max(0, Q - demand);
    const profit = price * sales - cost * Q - disposal * leftover + salvage * leftover;
    
    profits.push(profit);
    
    if (i < 100) {
      scenarios.push({
        iteration: i + 1,
        demand,
        sales,
        leftover,
        profit: parseFloat(profit.toFixed(2))
      });
    }
  }
  
  // Calculate statistics
  profits.sort((a, b) => a - b);
  const sum = profits.reduce((a, b) => a + b, 0);
  const mean = sum / iterations;
  const variance = profits.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / iterations;
  const stdDev = Math.sqrt(variance);
  
  const percentiles = {
    p5: profits[Math.floor(iterations * 0.05)],
    p25: profits[Math.floor(iterations * 0.25)],
    p50: profits[Math.floor(iterations * 0.50)],
    p75: profits[Math.floor(iterations * 0.75)],
    p95: profits[Math.floor(iterations * 0.95)]
  };
  
  return {
    avgProfit: mean,
    stdDev,
    percentiles,
    scenarios
  };
}

// API Routes

/**
 * GET /api/health - Health check
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

/**
 * POST /api/optimize - Calculate optimal order quantity
 */
app.post('/api/optimize', async (req, res) => {
  try {
    const params = req.body;
    
    // Validate parameters
    if (!params.price || !params.cost || params.disposal === undefined) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    // Calculate optimal Q
    const optimization = calculateOptimalQ(params);
    const expectedProfit = calculateExpectedProfit(optimization.optimalQ, params);
    
    const result = {
      ...optimization,
      expectedProfit: parseFloat(expectedProfit.toFixed(2))
    };
    
    res.json(result);
  } catch (error) {
    console.error('Optimization error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/simulate - Run Monte Carlo simulation
 */
app.post('/api/simulate', async (req, res) => {
  try {
    const { orderQuantity, parameters, iterations = 10000 } = req.body;
    
    if (!orderQuantity || !parameters) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    const simulation = runMonteCarloSimulation(orderQuantity, parameters, iterations);
    
    res.json({
      orderQuantity,
      iterations,
      ...simulation
    });
  } catch (error) {
    console.error('Simulation error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/profit-curve - Generate profit curve data
 */
app.get('/api/profit-curve', async (req, res) => {
  try {
    const {
      qMin = 0,
      qMax = 200,
      step = 2,
      price,
      cost,
      disposal,
      salvage = 0,
      demandMean,
      demandStd,
      distribution = 'normal',
      poissonLambda
    } = req.query;
    
    const params = {
      price: parseFloat(price),
      cost: parseFloat(cost),
      disposal: parseFloat(disposal),
      salvage: parseFloat(salvage),
      demandMean: parseFloat(demandMean),
      demandStd: parseFloat(demandStd),
      distribution,
      poissonLambda: parseFloat(poissonLambda)
    };
    
    const curve = [];
    
    for (let q = parseInt(qMin); q <= parseInt(qMax); q += parseInt(step)) {
      const profit = calculateExpectedProfit(q, params);
      curve.push({
        q,
        profit: parseFloat(profit.toFixed(2))
      });
    }
    
    res.json(curve);
  } catch (error) {
    console.error('Profit curve error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/scenarios - Save scenario to database
 */
app.post('/api/scenarios', async (req, res) => {
  try {
    const scenario = new Scenario(req.body);
    await scenario.save();
    
    res.status(201).json({
      message: 'Scenario saved successfully',
      id: scenario._id
    });
  } catch (error) {
    console.error('Save scenario error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/scenarios - Retrieve scenarios
 */
app.get('/api/scenarios', async (req, res) => {
  try {
    const { limit = 50, skip = 0 } = req.query;
    
    const scenarios = await Scenario
      .find()
      .sort({ timestamp: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip));
    
    const total = await Scenario.countDocuments();
    
    res.json({
      scenarios,
      total,
      limit: parseInt(limit),
      skip: parseInt(skip)
    });
  } catch (error) {
    console.error('Retrieve scenarios error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/scenarios/:id - Get specific scenario
 */
app.get('/api/scenarios/:id', async (req, res) => {
  try {
    const scenario = await Scenario.findById(req.params.id);
    
    if (!scenario) {
      return res.status(404).json({ error: 'Scenario not found' });
    }
    
    res.json(scenario);
  } catch (error) {
    console.error('Get scenario error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * DELETE /api/scenarios/:id - Delete scenario
 */
app.delete('/api/scenarios/:id', async (req, res) => {
  try {
    const result = await Scenario.findByIdAndDelete(req.params.id);
    
    if (!result) {
      return res.status(404).json({ error: 'Scenario not found' });
    }
    
    res.json({ message: 'Scenario deleted successfully' });
  } catch (error) {
    console.error('Delete scenario error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/compare - Compare multiple order quantities
 */
app.post('/api/compare', async (req, res) => {
  try {
    const { quantities, parameters } = req.body;
    
    if (!quantities || !Array.isArray(quantities) || !parameters) {
      return res.status(400).json({ error: 'Invalid request format' });
    }
    
    const comparisons = quantities.map(q => ({
      quantity: q,
      expectedProfit: calculateExpectedProfit(q, parameters),
      simulation: runMonteCarloSimulation(q, parameters, 1000)
    }));
    
    res.json({ comparisons });
  } catch (error) {
    console.error('Comparison error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Error handling middleware
 */
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     NEWSVENDOR PROBLEM - BACKEND API SERVER              ║
║                                                          ║
║     Server running on port ${PORT}                          ║
║     MongoDB: ${MONGODB_URI.substring(0, 40)}...║
║                                                          ║
║     Endpoints:                                           ║
║     - POST /api/optimize                                 ║
║     - POST /api/simulate                                 ║
║     - GET  /api/profit-curve                             ║
║     - POST /api/scenarios                                ║
║     - GET  /api/scenarios                                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
  `);
});

module.exports = app;