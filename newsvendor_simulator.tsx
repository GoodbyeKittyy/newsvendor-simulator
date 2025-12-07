import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, Area, AreaChart } from 'recharts';
import { TrendingUp, AlertCircle, DollarSign, Package, Trash2 } from 'lucide-react';

const NewsvendorSimulator = () => {
  const [params, setParams] = useState({
    price: 5,
    cost: 2,
    disposal: 0.5,
    salvage: 0,
    demandMean: 100,
    demandStd: 20,
    distribution: 'normal',
    poissonLambda: 100,
    leadTime: 0
  });

  const [orderQty, setOrderQty] = useState(100);
  const [simResults, setSimResults] = useState(null);
  const [optimalQ, setOptimalQ] = useState(0);
  const [profitCurve, setProfitCurve] = useState([]);
  const [simulation, setSimulation] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const normalCDF = (x, mean, std) => {
    const z = (x - mean) / std;
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - p : p;
  };

  const normalPDF = (x, mean, std) => {
    return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
  };

  const poissonPMF = (k, lambda) => {
    if (k < 0) return 0;
    return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
  };

  const factorial = (n) => {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) result *= i;
    return result;
  };

  const generateDemand = (distribution, mean, std, lambda) => {
    if (distribution === 'normal') {
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return Math.max(0, Math.round(mean + z * std));
    } else if (distribution === 'poisson') {
      let L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= Math.random();
      } while (p > L);
      return k - 1;
    }
    return mean;
  };

  const calculateOptimalQ = () => {
    const { price, cost, disposal, salvage } = params;
    const cu = price - cost; // underage cost
    const co = cost + disposal - salvage; // overage cost
    const criticalFractile = cu / (cu + co);

    let optimal = 0;
    if (params.distribution === 'normal') {
      // Find Q where F(Q) = critical fractile
      for (let q = 0; q < params.demandMean * 3; q++) {
        const cdf = normalCDF(q, params.demandMean, params.demandStd);
        if (cdf >= criticalFractile) {
          optimal = q;
          break;
        }
      }
    } else if (params.distribution === 'poisson') {
      let cumProb = 0;
      for (let q = 0; q < params.poissonLambda * 3; q++) {
        cumProb += poissonPMF(q, params.poissonLambda);
        if (cumProb >= criticalFractile) {
          optimal = q;
          break;
        }
      }
    }
    return optimal;
  };

  const calculateExpectedProfit = (q) => {
    const { price, cost, disposal, salvage, demandMean, demandStd, distribution, poissonLambda } = params;
    
    let expectedSales = 0;
    let expectedLeftover = 0;

    if (distribution === 'normal') {
      for (let d = 0; d <= demandMean * 3; d++) {
        const prob = normalPDF(d, demandMean, demandStd);
        expectedSales += Math.min(d, q) * prob;
        expectedLeftover += Math.max(0, q - d) * prob;
      }
    } else if (distribution === 'poisson') {
      for (let d = 0; d <= poissonLambda * 3; d++) {
        const prob = poissonPMF(d, poissonLambda);
        expectedSales += Math.min(d, q) * prob;
        expectedLeftover += Math.max(0, q - d) * prob;
      }
    }

    const revenue = price * expectedSales;
    const productionCost = cost * q;
    const disposalCost = disposal * expectedLeftover;
    const salvageRevenue = salvage * expectedLeftover;

    return revenue - productionCost - disposalCost + salvageRevenue;
  };

  const runMonteCarloSimulation = (q, iterations = 1000) => {
    const { price, cost, disposal, salvage, demandMean, demandStd, distribution, poissonLambda } = params;
    let totalProfit = 0;
    const results = [];

    for (let i = 0; i < iterations; i++) {
      const demand = generateDemand(
        distribution,
        demandMean,
        demandStd,
        poissonLambda
      );
      const sales = Math.min(demand, q);
      const leftover = Math.max(0, q - demand);
      const profit = price * sales - cost * q - disposal * leftover + salvage * leftover;
      totalProfit += profit;
      
      if (i < 50) {
        results.push({ iteration: i + 1, demand, sales, leftover, profit });
      }
    }

    return {
      avgProfit: totalProfit / iterations,
      simulations: results
    };
  };

  useEffect(() => {
    const optimal = calculateOptimalQ();
    setOptimalQ(optimal);

    const curve = [];
    const start = Math.max(0, optimal - 50);
    const end = optimal + 50;
    
    for (let q = start; q <= end; q += 2) {
      const profit = calculateExpectedProfit(q);
      curve.push({ q, profit: profit.toFixed(2) });
    }
    setProfitCurve(curve);

    const results = runMonteCarloSimulation(orderQty, 1000);
    setSimResults(results);
    setSimulation(results.simulations);
  }, [params, orderQty]);

  const currentProfit = calculateExpectedProfit(orderQty);
  const optimalProfit = calculateExpectedProfit(optimalQ);
  const efficiency = ((currentProfit / optimalProfit) * 100).toFixed(1);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom, #f5f5f0 0%, #e8e8e0 100%)',
      fontFamily: '"Courier New", Courier, monospace',
      padding: '20px'
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        background: '#fafaf5',
        border: '3px solid #333',
        boxShadow: '0 8px 16px rgba(0,0,0,0.3)'
      }}>
        {/* Header */}
        <div style={{
          background: '#333',
          color: '#fff',
          padding: '20px',
          borderBottom: '6px double #666'
        }}>
          <div style={{
            fontSize: '48px',
            fontWeight: 'bold',
            fontFamily: '"Times New Roman", Times, serif',
            textAlign: 'center',
            letterSpacing: '2px',
            textTransform: 'uppercase'
          }}>
            The Newsvendor Chronicle
          </div>
          <div style={{
            textAlign: 'center',
            marginTop: '8px',
            fontSize: '14px',
            letterSpacing: '3px',
            fontStyle: 'italic'
          }}>
            Est. 1888 • Perishable Goods & Stochastic Optimization • Vol. CXXXVI
          </div>
        </div>

        <div style={{ padding: '30px' }}>
          {/* Main Headline */}
          <div style={{
            borderBottom: '3px solid #333',
            borderTop: '3px solid #333',
            padding: '15px 0',
            marginBottom: '30px'
          }}>
            <h1 style={{
              fontSize: '36px',
              fontFamily: '"Times New Roman", Times, serif',
              textAlign: 'center',
              margin: 0,
              fontWeight: 'bold'
            }}>
              BAKERY CRISIS: HOW MANY CROISSANTS TO BAKE?
            </h1>
            <p style={{
              textAlign: 'center',
              fontSize: '16px',
              marginTop: '10px',
              fontStyle: 'italic'
            }}>
              Mathematical Solution to Age-Old Dilemma of Uncertain Demand
            </p>
          </div>

          {/* Control Panel */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '30px',
            marginBottom: '30px'
          }}>
            <div style={{
              border: '2px solid #333',
              padding: '20px',
              background: '#fff'
            }}>
              <h2 style={{
                fontSize: '24px',
                fontFamily: '"Times New Roman", Times, serif',
                borderBottom: '2px solid #333',
                paddingBottom: '10px',
                marginTop: 0
              }}>
                PARAMETERS
              </h2>
              
              <div style={{ marginTop: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Selling Price (p): ${params.price}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="0.5"
                  value={params.price}
                  onChange={(e) => setParams({...params, price: parseFloat(e.target.value)})}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginTop: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Production Cost (c): ${params.cost}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="15"
                  step="0.5"
                  value={params.cost}
                  onChange={(e) => setParams({...params, cost: parseFloat(e.target.value)})}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginTop: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Disposal Cost (h): ${params.disposal}
                </label>
                <input
                  type="range"
                  min="0"
                  max="5"
                  step="0.1"
                  value={params.disposal}
                  onChange={(e) => setParams({...params, disposal: parseFloat(e.target.value)})}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginTop: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Distribution Type:
                </label>
                <select
                  value={params.distribution}
                  onChange={(e) => setParams({...params, distribution: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '8px',
                    fontFamily: 'inherit',
                    border: '2px solid #333'
                  }}
                >
                  <option value="normal">Normal Distribution</option>
                  <option value="poisson">Poisson Distribution</option>
                </select>
              </div>

              {params.distribution === 'normal' && (
                <>
                  <div style={{ marginTop: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                      Demand Mean (μ): {params.demandMean}
                    </label>
                    <input
                      type="range"
                      min="20"
                      max="300"
                      value={params.demandMean}
                      onChange={(e) => setParams({...params, demandMean: parseInt(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div style={{ marginTop: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                      Demand Std Dev (σ): {params.demandStd}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      value={params.demandStd}
                      onChange={(e) => setParams({...params, demandStd: parseInt(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>
                </>
              )}

              {params.distribution === 'poisson' && (
                <div style={{ marginTop: '15px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    Lambda (λ): {params.poissonLambda}
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="300"
                    value={params.poissonLambda}
                    onChange={(e) => setParams({...params, poissonLambda: parseInt(e.target.value)})}
                    style={{ width: '100%' }}
                  />
                </div>
              )}

              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{
                  marginTop: '15px',
                  padding: '10px',
                  width: '100%',
                  background: '#333',
                  color: '#fff',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: 'bold'
                }}
              >
                {showAdvanced ? 'HIDE' : 'SHOW'} ADVANCED OPTIONS
              </button>

              {showAdvanced && (
                <>
                  <div style={{ marginTop: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                      Salvage Value: ${params.salvage}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="5"
                      step="0.1"
                      value={params.salvage}
                      onChange={(e) => setParams({...params, salvage: parseFloat(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div style={{ marginTop: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                      Lead Time (days): {params.leadTime}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="7"
                      value={params.leadTime}
                      onChange={(e) => setParams({...params, leadTime: parseInt(e.target.value)})}
                      style={{ width: '100%' }}
                    />
                  </div>
                </>
              )}
            </div>

            <div style={{
              border: '2px solid #333',
              padding: '20px',
              background: '#fff'
            }}>
              <h2 style={{
                fontSize: '24px',
                fontFamily: '"Times New Roman", Times, serif',
                borderBottom: '2px solid #333',
                paddingBottom: '10px',
                marginTop: 0
              }}>
                ORDER QUANTITY
              </h2>

              <div style={{ marginTop: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                  Your Order Quantity (Q): {orderQty} units
                </label>
                <input
                  type="range"
                  min="0"
                  max={params.demandMean * 2}
                  value={orderQty}
                  onChange={(e) => setOrderQty(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{
                marginTop: '25px',
                padding: '20px',
                background: '#f0f0e8',
                border: '2px solid #666'
              }}>
                <h3 style={{
                  fontSize: '20px',
                  margin: '0 0 15px 0',
                  textAlign: 'center',
                  borderBottom: '1px solid #666',
                  paddingBottom: '10px'
                }}>
                  CRITICAL FRACTILE SOLUTION
                </h3>
                
                <div style={{ fontSize: '14px', lineHeight: '1.8' }}>
                  <p style={{ margin: '5px 0' }}>
                    <strong>Optimal Q*:</strong> {optimalQ} units
                  </p>
                  <p style={{ margin: '5px 0' }}>
                    <strong>Critical Fractile:</strong> {(
                      (params.price - params.cost) / 
                      (params.price - params.cost + params.cost + params.disposal - params.salvage)
                    ).toFixed(3)}
                  </p>
                  <p style={{ margin: '5px 0' }}>
                    <strong>Underage Cost (Cu):</strong> ${(params.price - params.cost).toFixed(2)}
                  </p>
                  <p style={{ margin: '5px 0' }}>
                    <strong>Overage Cost (Co):</strong> ${(params.cost + params.disposal - params.salvage).toFixed(2)}
                  </p>
                </div>
              </div>

              <div style={{
                marginTop: '20px',
                padding: '15px',
                background: currentProfit < optimalProfit * 0.9 ? '#ffe6e6' : '#e6f7e6',
                border: '2px solid #333'
              }}>
                <h3 style={{ fontSize: '18px', margin: '0 0 10px 0' }}>
                  PERFORMANCE REPORT
                </h3>
                <p style={{ margin: '5px 0' }}>
                  <strong>Your Expected Profit:</strong> ${currentProfit.toFixed(2)}
                </p>
                <p style={{ margin: '5px 0' }}>
                  <strong>Optimal Profit:</strong> ${optimalProfit.toFixed(2)}
                </p>
                <p style={{ margin: '5px 0' }}>
                  <strong>Efficiency:</strong> {efficiency}%
                </p>
              </div>

              <button
                onClick={() => setOrderQty(optimalQ)}
                style={{
                  marginTop: '15px',
                  padding: '12px',
                  width: '100%',
                  background: '#2a5a2a',
                  color: '#fff',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: 'bold',
                  fontSize: '16px'
                }}
              >
                SET TO OPTIMAL Q*
              </button>
            </div>
          </div>

          {/* Profit Curve Chart */}
          <div style={{
            border: '2px solid #333',
            padding: '20px',
            background: '#fff',
            marginBottom: '30px'
          }}>
            <h2 style={{
              fontSize: '24px',
              fontFamily: '"Times New Roman", Times, serif',
              borderBottom: '2px solid #333',
              paddingBottom: '10px',
              marginTop: 0,
              textAlign: 'center'
            }}>
              EXPECTED PROFIT CURVE: E[π(Q)]
            </h2>
            <LineChart width={1300} height={400} data={profitCurve} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
              <XAxis 
                dataKey="q" 
                label={{ value: 'Order Quantity (Q)', position: 'insideBottom', offset: -10 }}
                stroke="#333"
              />
              <YAxis 
                label={{ value: 'Expected Profit ($)', angle: -90, position: 'insideLeft' }}
                stroke="#333"
              />
              <Tooltip 
                contentStyle={{ 
                  background: '#fafaf5', 
                  border: '2px solid #333',
                  fontFamily: 'inherit'
                }}
              />
              <Legend />
              <ReferenceLine 
                x={optimalQ} 
                stroke="#2a5a2a" 
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: `Optimal Q* = ${optimalQ}`, position: 'top' }}
              />
              <ReferenceLine 
                x={orderQty} 
                stroke="#8b0000" 
                strokeWidth={2}
                label={{ value: `Your Q = ${orderQty}`, position: 'top' }}
              />
              <Line 
                type="monotone" 
                dataKey="profit" 
                stroke="#333" 
                strokeWidth={3}
                dot={false}
                name="Expected Profit"
              />
            </LineChart>
            <p style={{ 
              textAlign: 'center', 
              marginTop: '15px',
              fontStyle: 'italic',
              fontSize: '14px'
            }}>
              Note the asymmetric costs: understocking (left of Q*) vs. overstocking (right of Q*)
            </p>
          </div>

          {/* Monte Carlo Results */}
          <div style={{
            border: '2px solid #333',
            padding: '20px',
            background: '#fff'
          }}>
            <h2 style={{
              fontSize: '24px',
              fontFamily: '"Times New Roman", Times, serif',
              borderBottom: '2px solid #333',
              paddingBottom: '10px',
              marginTop: 0,
              textAlign: 'center'
            }}>
              MONTE CARLO SIMULATION (1,000 SCENARIOS)
            </h2>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '20px',
              marginTop: '20px'
            }}>
              <div style={{
                padding: '15px',
                background: '#f0f0e8',
                border: '2px solid #666',
                textAlign: 'center'
              }}>
                <DollarSign size={32} style={{ margin: '0 auto 10px' }} />
                <h3 style={{ margin: '5px 0', fontSize: '18px' }}>Avg Profit</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: '5px 0' }}>
                  ${simResults?.avgProfit.toFixed(2)}
                </p>
              </div>
              
              <div style={{
                padding: '15px',
                background: '#f0f0e8',
                border: '2px solid #666',
                textAlign: 'center'
              }}>
                <Package size={32} style={{ margin: '0 auto 10px' }} />
                <h3 style={{ margin: '5px 0', fontSize: '18px' }}>Order Qty</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: '5px 0' }}>
                  {orderQty} units
                </p>
              </div>
              
              <div style={{
                padding: '15px',
                background: '#f0f0e8',
                border: '2px solid #666',
                textAlign: 'center'
              }}>
                <TrendingUp size={32} style={{ margin: '0 auto 10px' }} />
                <h3 style={{ margin: '5px 0', fontSize: '18px' }}>Efficiency</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: '5px 0' }}>
                  {efficiency}%
                </p>
              </div>
            </div>

            <div style={{
              marginTop: '25px',
              maxHeight: '300px',
              overflowY: 'auto',
              border: '2px solid #333',
              background: '#fafaf5'
            }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '12px'
              }}>
                <thead>
                  <tr style={{ background: '#333', color: '#fff' }}>
                    <th style={{ padding: '10px', borderBottom: '2px solid #333' }}>Trial</th>
                    <th style={{ padding: '10px', borderBottom: '2px solid #333' }}>Demand</th>
                    <th style={{ padding: '10px', borderBottom: '2px solid #333' }}>Sales</th>
                    <th style={{ padding: '10px', borderBottom: '2px solid #333' }}>Leftover</th>
                    <th style={{ padding: '10px', borderBottom: '2px solid #333' }}>Profit</th>
                  </tr>
                </thead>
                <tbody>
                  {simulation.map((sim, idx) => (
                    <tr key={idx} style={{
                      background: idx % 2 === 0 ? '#fff' : '#f0f0e8'
                    }}>
                      <td style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid #ccc' }}>
                        {sim.iteration}
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid #ccc' }}>
                        {sim.demand}
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid #ccc' }}>
                        {sim.sales}
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid #ccc' }}>
                        {sim.leftover}
                      </td>
                      <td style={{
                        padding: '8px',
                        textAlign: 'center',
                        borderBottom: '1px solid #ccc',
                        color: sim.profit < 0 ? '#8b0000' : '#2a5a2a',
                        fontWeight: 'bold'
                      }}>
                        ${sim.profit.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Footer */}
          <div style={{
            marginTop: '30px',
            padding: '20px',
            borderTop: '3px double #333',
            textAlign: 'center',
            fontSize: '12px',
            fontStyle: 'italic'
          }}>
            <p>
              "The optimal policy intentionally stocks out sometimes - this is not a failure, but mathematical wisdom."
            </p>
            <p style={{ marginTop: '10px' }}>
              Formula: F(Q*) = (p-c)/(p-c+h) where p=price, c=cost, h=disposal cost
            </p>
            <p style={{ marginTop: '10px' }}>
              E[π(Q)] = p·E[min(D,Q)] - cQ - h·E[max(0,Q-D)]
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsvendorSimulator;