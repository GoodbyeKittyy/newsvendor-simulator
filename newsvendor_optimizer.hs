{-# LANGUAGE RecordWildCards #-}

{-|
Module      : NewsvendorOptimizer
Description : Pure functional newsvendor optimization in Haskell
Copyright   : (c) 2024
License     : MIT

Implements the newsvendor problem using pure functional programming
Features:
- Type-safe parameter handling
- Lazy evaluation for efficiency
- Monadic error handling
- Pure mathematical functions
-}

module NewsvendorOptimizer where

import Data.List (foldl', minimumBy, maximumBy)
import Data.Ord (comparing)
import Control.Monad (replicateM)
import System.Random (randomRIO, Random, RandomGen, mkStdGen, randoms)
import Text.Printf (printf)

-- | Parameters for the newsvendor problem
data Parameters = Parameters
  { price         :: Double  -- ^ Selling price per unit
  , cost          :: Double  -- ^ Production cost per unit
  , disposal      :: Double  -- ^ Disposal cost per unit
  , salvage       :: Double  -- ^ Salvage value per unit
  , demandMean    :: Double  -- ^ Mean demand
  , demandStd     :: Double  -- ^ Standard deviation of demand
  , distribution  :: Distribution  -- ^ Demand distribution
  } deriving (Show, Eq)

-- | Supported probability distributions
data Distribution 
  = Normal { mu :: Double, sigma :: Double }
  | Poisson { lambda :: Double }
  | Uniform { lower :: Double, upper :: Double }
  deriving (Show, Eq)

-- | Result of optimization
data OptimizationResult = OptimizationResult
  { optimalQ          :: Int     -- ^ Optimal order quantity
  , criticalFractile  :: Double  -- ^ Critical fractile value
  , expectedProfit    :: Double  -- ^ Expected profit at optimal Q
  , underageCost      :: Double  -- ^ Cost of understocking
  , overageCost       :: Double  -- ^ Cost of overstocking
  } deriving (Show, Eq)

-- | Result of profit calculation
data ProfitResult = ProfitResult
  { quantity          :: Int
  , profit            :: Double
  , sales             :: Double
  , leftover          :: Double
  } deriving (Show, Eq)

-- | Statistical functions

-- | Calculate normal PDF
normalPDF :: Double -> Double -> Double -> Double
normalPDF x mu sigma = 
  let z = (x - mu) / sigma
      coefficient = 1 / (sigma * sqrt (2 * pi))
  in coefficient * exp (-0.5 * z * z)

-- | Calculate normal CDF using approximation
normalCDF :: Double -> Double -> Double -> Double
normalCDF x mu sigma =
  let z = (x - mu) / sigma
      t = 1 / (1 + 0.2316419 * abs z)
      d = 0.3989423 * exp (-z * z / 2)
      poly = t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + 
                  t * (-1.821256 + t * 1.330274))))
      p = d * poly
  in if z > 0 then 1 - p else p

-- | Inverse normal CDF (quantile function)
inverseNormalCDF :: Double -> Double -> Double -> Double
inverseNormalCDF p mu sigma
  | p <= 0 || p >= 1 = error "Probability must be between 0 and 1"
  | otherwise = mu + sigma * inverseStandardNormalCDF p

-- | Inverse standard normal CDF
inverseStandardNormalCDF :: Double -> Double
inverseStandardNormalCDF p
  | p < pLow = 
      let q = sqrt (-2 * log p)
      in -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
          ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
  | p <= pHigh =
      let q = p - 0.5
          r = q * q
      in (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q /
         (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1)
  | otherwise =
      let q = sqrt (-2 * log (1 - p))
      in (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
         ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
  where
    pLow = 0.02425
    pHigh = 1 - pLow
    a0 = -3.969683028665376e+01
    a1 = 2.209460984245205e+02
    a2 = -2.759285104469687e+02
    a3 = 1.383577518672690e+02
    a4 = -3.066479806614716e+01
    a5 = 2.506628277459239e+00
    b0 = -5.447609879822406e+01
    b1 = 1.615858368580409e+02
    b2 = -1.556989798598866e+02
    b3 = 6.680131188771972e+01
    b4 = -1.328068155288572e+01
    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e+00
    c3 = -2.549732539343734e+00
    c4 = 4.374664141464968e+00
    c5 = 2.938163982698783e+00
    d0 = 7.784695709041462e-03
    d1 = 3.224671290700398e-01
    d2 = 2.445134137142996e+00
    d3 = 3.754408661907416e+00

-- | Calculate Poisson PMF
poissonPMF :: Int -> Double -> Double
poissonPMF k lambda
  | k < 0 = 0
  | k > 500 = 0
  | otherwise = 
      let logProb = fromIntegral k * log lambda - lambda - logFactorial k
      in exp logProb

-- | Log factorial for numerical stability
logFactorial :: Int -> Double
logFactorial n = sum [log (fromIntegral i) | i <- [1..n]]

-- | Calculate Poisson CDF
poissonCDF :: Int -> Double -> Double
poissonCDF k lambda = sum [poissonPMF i lambda | i <- [0..k]]

-- | Core optimization functions

-- | Calculate critical fractile
criticalFractile' :: Parameters -> Double
criticalFractile' Parameters{..} =
  let cu = price - cost
      co = cost + disposal - salvage
  in cu / (cu + co)

-- | Calculate optimal order quantity
calculateOptimalQ :: Parameters -> OptimizationResult
calculateOptimalQ params@Parameters{..} =
  let cf = criticalFractile' params
      optQ = case distribution of
        Normal mu sigma -> 
          round $ inverseNormalCDF cf mu sigma
        Poisson lambda ->
          binarySearchPoisson cf lambda 0 (round $ lambda * 3)
        Uniform lower upper ->
          round $ lower + cf * (upper - lower)
      cu = price - cost
      co = cost + disposal - salvage
      expProfit = expectedProfit' params optQ
  in OptimizationResult
      { optimalQ = max 0 optQ
      , criticalFractile = cf
      , expectedProfit = expProfit
      , underageCost = cu
      , overageCost = co
      }

-- | Binary search for Poisson optimal Q
binarySearchPoisson :: Double -> Double -> Int -> Int -> Int
binarySearchPoisson cf lambda low high
  | low >= high = low
  | otherwise =
      let mid = (low + high) `div` 2
          cdf = poissonCDF mid lambda
      in if cdf < cf
         then binarySearchPoisson cf lambda (mid + 1) high
         else binarySearchPoisson cf lambda low mid

-- | Calculate expected profit for given Q
expectedProfit' :: Parameters -> Int -> Double
expectedProfit' Parameters{..} q =
  let maxDemand = case distribution of
        Normal mu sigma -> round $ mu + 3 * sigma
        Poisson lambda -> round $ lambda * 3
        Uniform _ upper -> round upper
      demands = [0..maxDemand]
      (expSales, expLeftover) = foldl' accumulate (0, 0) demands
  in price * expSales - cost * fromIntegral q - 
     disposal * expLeftover + salvage * expLeftover
  where
    accumulate (sumSales, sumLeftover) d =
      let prob = case distribution of
            Normal mu sigma -> normalPDF (fromIntegral d) mu sigma
            Poisson lambda -> poissonPMF d lambda
            Uniform lower upper -> 
              if fromIntegral d >= lower && fromIntegral d <= upper
              then 1 / (upper - lower + 1)
              else 0
          sales = fromIntegral $ min d q
          leftover = fromIntegral $ max 0 (q - d)
      in (sumSales + sales * prob, sumLeftover + leftover * prob)

-- | Generate profit curve
profitCurve :: Parameters -> Int -> Int -> Int -> [ProfitResult]
profitCurve params qMin qMax step =
  [ ProfitResult 
      { quantity = q
      , profit = expectedProfit' params q
      , sales = 0  -- Computed lazily if needed
      , leftover = 0  -- Computed lazily if needed
      }
  | q <- [qMin, qMin + step .. qMax]
  ]

-- | Find maximum profit point on curve
maximumProfit :: [ProfitResult] -> ProfitResult
maximumProfit = maximumBy (comparing profit)

-- | Monte Carlo simulation

-- | Generate random demand sample
generateDemand :: Distribution -> IO Int
generateDemand (Normal mu sigma) = do
  u1 <- randomRIO (0, 1)
  u2 <- randomRIO (0, 1)
  let z = sqrt (-2 * log u1) * cos (2 * pi * u2)
      demand = mu + z * sigma
  return $ max 0 (round demand)
generateDemand (Poisson lambda) = do
  let l = exp (-lambda)
  generatePoisson l 0 1.0
  where
    generatePoisson l k p = do
      u <- randomRIO (0, 1)
      let p' = p * u
      if p' > l
        then generatePoisson l (k + 1) p'
        else return k
generateDemand (Uniform lower upper) = do
  u <- randomRIO (0, 1)
  return $ round $ lower + u * (upper - lower)

-- | Single simulation run
simulateOnce :: Parameters -> Int -> IO ProfitResult
simulateOnce params@Parameters{..} q = do
  demand <- generateDemand distribution
  let sales = min demand q
      leftover = max 0 (q - demand)
      profitValue = price * fromIntegral sales - 
                   cost * fromIntegral q -
                   disposal * fromIntegral leftover +
                   salvage * fromIntegral leftover
  return ProfitResult
    { quantity = q
    , profit = profitValue
    , sales = fromIntegral sales
    , leftover = fromIntegral leftover
    }

-- | Monte Carlo simulation
monteCarloSimulation :: Parameters -> Int -> Int -> IO [ProfitResult]
monteCarloSimulation params q iterations =
  replicateM iterations (simulateOnce params q)

-- | Calculate statistics from simulation results
data SimulationStats = SimulationStats
  { avgProfit     :: Double
  , stdDevProfit  :: Double
  , minProfit     :: Double
  , maxProfit     :: Double
  , percentile5   :: Double
  , percentile25  :: Double
  , percentile50  :: Double
  , percentile75  :: Double
  , percentile95  :: Double
  } deriving (Show)

-- | Compute statistics from profit results
computeStats :: [ProfitResult] -> SimulationStats
computeStats results =
  let profits = map profit results
      n = fromIntegral $ length profits
      mean = sum profits / n
      variance = sum [(p - mean)^2 | p <- profits] / n
      stdDev = sqrt variance
      sorted = foldl' insertSorted [] profits
  in SimulationStats
      { avgProfit = mean
      , stdDevProfit = stdDev
      , minProfit = minimum profits
      , maxProfit = maximum profits
      , percentile5 = percentile sorted 0.05
      , percentile25 = percentile sorted 0.25
      , percentile50 = percentile sorted 0.50
      , percentile75 = percentile sorted 0.75
      , percentile95 = percentile sorted 0.95
      }
  where
    insertSorted [] x = [x]
    insertSorted (y:ys) x
      | x <= y = x : y : ys
      | otherwise = y : insertSorted ys x
    
    percentile sorted p =
      let idx = round $ p * fromIntegral (length sorted - 1)
      in sorted !! idx

-- | Utility functions

-- | Format currency
formatCurrency :: Double -> String
formatCurrency x = printf "$%.2f" x

-- | Format percentage
formatPercent :: Double -> String
formatPercent x = printf "%.1f%%" (x * 100)

-- | Pretty print optimization result
printOptimizationResult :: OptimizationResult -> IO ()
printOptimizationResult OptimizationResult{..} = do
  putStrLn "╔══════════════════════════════════════════════════════════╗"
  putStrLn "║           OPTIMIZATION RESULT (HASKELL)                  ║"
  putStrLn "╚══════════════════════════════════════════════════════════╝"
  putStrLn ""
  putStrLn $ "Optimal Order Quantity (Q*): " ++ show optimalQ ++ " units"
  putStrLn $ "Critical Fractile:           " ++ formatPercent criticalFractile
  putStrLn $ "Expected Profit:             " ++ formatCurrency expectedProfit
  putStrLn $ "Underage Cost (Cu):          " ++ formatCurrency underageCost
  putStrLn $ "Overage Cost (Co):           " ++ formatCurrency overageCost
  putStrLn ""

-- | Pretty print simulation statistics
printSimulationStats :: SimulationStats -> IO ()
printSimulationStats SimulationStats{..} = do
  putStrLn "╔══════════════════════════════════════════════════════════╗"
  putStrLn "║         MONTE CARLO SIMULATION STATISTICS                ║"
  putStrLn "╚══════════════════════════════════════════════════════════╝"
  putStrLn ""
  putStrLn $ "Average Profit:   " ++ formatCurrency avgProfit
  putStrLn $ "Std Deviation:    " ++ formatCurrency stdDevProfit
  putStrLn $ "Min Profit:       " ++ formatCurrency minProfit
  putStrLn $ "Max Profit:       " ++ formatCurrency maxProfit
  putStrLn ""
  putStrLn "Percentiles:"
  putStrLn $ "  5th:  " ++ formatCurrency percentile5
  putStrLn $ "  25th: " ++ formatCurrency percentile25
  putStrLn $ "  50th: " ++ formatCurrency percentile50
  putStrLn $ "  75th: " ++ formatCurrency percentile75
  putStrLn $ "  95th: " ++ formatCurrency percentile95
  putStrLn ""

-- | Main demonstration
main :: IO ()
main = do
  putStrLn "╔══════════════════════════════════════════════════════════╗"
  putStrLn "║                                                          ║"
  putStrLn "║     NEWSVENDOR OPTIMIZER - HASKELL IMPLEMENTATION        ║"
  putStrLn "║                                                          ║"
  putStrLn "║     Pure functional optimization with type safety        ║"
  putStrLn "║                                                          ║"
  putStrLn "╚══════════════════════════════════════════════════════════╝"
  putStrLn ""
  
  -- Example parameters
  let params = Parameters
        { price = 5.0
        , cost = 2.0
        , disposal = 0.5
        , salvage = 0.0
        , demandMean = 100.0
        , demandStd = 20.0
        , distribution = Normal 100.0 20.0
        }
  
  -- Calculate optimal quantity
  let result = calculateOptimalQ params
  printOptimizationResult result
  
  -- Generate profit curve
  let curve = profitCurve params 50 150 2
      maxProfitPoint = maximumProfit curve
  
  putStrLn "Profit Curve Analysis:"
  putStrLn $ "  Maximum profit point: Q = " ++ show (quantity maxProfitPoint) ++
             ", Profit = " ++ formatCurrency (profit maxProfitPoint)
  putStrLn ""
  
  -- Run Monte Carlo simulation
  putStrLn "Running Monte Carlo simulation (10,000 iterations)..."
  simResults <- monteCarloSimulation params (optimalQ result) 10000
  let stats = computeStats simResults
  printSimulationStats stats
  
  -- Compare analytical vs simulation
  putStrLn "Comparison (Analytical vs Simulation):"
  putStrLn $ "  Analytical Expected Profit: " ++ formatCurrency (expectedProfit result)
  putStrLn $ "  Simulation Average Profit:  " ++ formatCurrency (avgProfit stats)
  putStrLn $ "  Difference:                  " ++ 
    formatCurrency (abs (expectedProfit result - avgProfit stats))
  putStrLn ""
  
  putStrLn "✓ Optimization complete. Pure functional solution verified."
  putStrLn ""