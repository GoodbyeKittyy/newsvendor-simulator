/**
 * Newsvendor Forecasting Engine - Scala Implementation
 * 
 * Advanced demand forecasting using time series analysis and machine learning
 * Features:
 * - Exponential smoothing (Holt-Winters)
 * - Moving average and weighted moving average
 * - Seasonal decomposition
 * - Trend analysis
 * - ARIMA-inspired forecasting
 */

package newsvendor

import scala.math._
import scala.collection.mutable.{ArrayBuffer, Map => MutableMap}
import scala.util.Random

// Case classes for data structures

case class Parameters(
  price: Double,
  cost: Double,
  disposal: Double,
  salvage: Double = 0.0,
  demandMean: Double,
  demandStd: Double,
  distribution: String = "normal",
  poissonLambda: Double = 0.0
)

case class DemandData(
  timestamp: Long,
  value: Double,
  dayOfWeek: Int,
  weekOfYear: Int,
  month: Int
)

case class ForecastResult(
  forecast: Double,
  lower95: Double,
  upper95: Double,
  trend: Double,
  seasonal: Double,
  method: String
)

case class TimeSeriesComponents(
  trend: Seq[Double],
  seasonal: Seq[Double],
  residual: Seq[Double],
  seasonalPeriod: Int
)

// Main Forecaster class
class DemandForecaster {
  
  /**
   * Simple Moving Average
   */
  def movingAverage(data: Seq[Double], window: Int): Seq[Double] = {
    if (data.length < window) return Seq.fill(data.length)(data.sum / data.length)
    
    data.sliding(window).map(_.sum / window).toSeq
  }
  
  /**
   * Weighted Moving Average (more recent values have higher weight)
   */
  def weightedMovingAverage(data: Seq[Double], window: Int): Seq[Double] = {
    if (data.length < window) return Seq.fill(data.length)(data.sum / data.length)
    
    val weights = (1 to window).map(_.toDouble)
    val sumWeights = weights.sum
    
    data.sliding(window).map { w =>
      w.zip(weights).map { case (v, wt) => v * wt }.sum / sumWeights
    }.toSeq
  }
  
  /**
   * Exponential Smoothing (single exponential)
   */
  def exponentialSmoothing(data: Seq[Double], alpha: Double = 0.3): Seq[Double] = {
    if (data.isEmpty) return Seq.empty
    
    val smoothed = ArrayBuffer[Double](data.head)
    
    for (i <- 1 until data.length) {
      val forecast = alpha * data(i) + (1 - alpha) * smoothed(i - 1)
      smoothed += forecast
    }
    
    smoothed.toSeq
  }
  
  /**
   * Double Exponential Smoothing (Holt's method) for trend
   */
  def doubleExponentialSmoothing(
    data: Seq[Double],
    alpha: Double = 0.3,
    beta: Double = 0.1
  ): Seq[Double] = {
    if (data.length < 2) return data
    
    var level = data.head
    var trend = data(1) - data.head
    val forecasts = ArrayBuffer[Double](data.head)
    
    for (i <- 1 until data.length) {
      val prevLevel = level
      level = alpha * data(i) + (1 - alpha) * (level + trend)
      trend = beta * (level - prevLevel) + (1 - beta) * trend
      forecasts += level + trend
    }
    
    forecasts.toSeq
  }
  
  /**
   * Triple Exponential Smoothing (Holt-Winters) for seasonal data
   */
  def holtWinters(
    data: Seq[Double],
    seasonLength: Int,
    alpha: Double = 0.3,
    beta: Double = 0.1,
    gamma: Double = 0.1
  ): ForecastResult = {
    if (data.length < seasonLength * 2) {
      return ForecastResult(
        forecast = data.lastOption.getOrElse(0.0),
        lower95 = 0.0,
        upper95 = 0.0,
        trend = 0.0,
        seasonal = 1.0,
        method = "insufficient_data"
      )
    }
    
    // Initialize components
    var level = data.take(seasonLength).sum / seasonLength
    var trend = (data.slice(seasonLength, seasonLength * 2).sum / seasonLength - level) / seasonLength
    val seasonal = MutableMap[Int, Double]()
    
    // Initialize seasonal factors
    for (i <- 0 until seasonLength) {
      val seasonalValues = (0 until data.length / seasonLength)
        .map(j => data(i + j * seasonLength))
      seasonal(i) = seasonalValues.sum / seasonalValues.length / level
    }
    
    // Update components
    for (i <- data.indices) {
      val seasonalIndex = i % seasonLength
      val prevLevel = level
      val prevTrend = trend
      
      level = alpha * (data(i) / seasonal(seasonalIndex)) + (1 - alpha) * (prevLevel + prevTrend)
      trend = beta * (level - prevLevel) + (1 - beta) * prevTrend
      seasonal(seasonalIndex) = gamma * (data(i) / level) + (1 - gamma) * seasonal(seasonalIndex)
    }
    
    // Forecast next period
    val forecastHorizon = 1
    val seasonalIndex = data.length % seasonLength
    val pointForecast = (level + forecastHorizon * trend) * seasonal(seasonalIndex)
    
    // Calculate prediction interval (simplified)
    val errors = data.sliding(2).map { case Seq(a, b) => abs(b - a) }.toSeq
    val mse = errors.map(e => e * e).sum / errors.length
    val stderr = sqrt(mse)
    
    ForecastResult(
      forecast = pointForecast,
      lower95 = pointForecast - 1.96 * stderr,
      upper95 = pointForecast + 1.96 * stderr,
      trend = trend,
      seasonal = seasonal(seasonalIndex),
      method = "holt_winters"
    )
  }
  
  /**
   * Seasonal Decomposition (additive model)
   */
  def seasonalDecompose(data: Seq[Double], period: Int): TimeSeriesComponents = {
    if (data.length < period * 2) {
      return TimeSeriesComponents(
        trend = data,
        seasonal = Seq.fill(data.length)(0.0),
        residual = Seq.fill(data.length)(0.0),
        seasonalPeriod = period
      )
    }
    
    // Calculate trend using centered moving average
    val trend = ArrayBuffer.fill[Double](data.length)(0.0)
    val halfPeriod = period / 2
    
    for (i <- halfPeriod until data.length - halfPeriod) {
      trend(i) = data.slice(i - halfPeriod, i + halfPeriod + 1).sum / (period + 1)
    }
    
    // Fill edges with nearest values
    for (i <- 0 until halfPeriod) trend(i) = trend(halfPeriod)
    for (i <- data.length - halfPeriod until data.length) trend(i) = trend(data.length - halfPeriod - 1)
    
    // Calculate detrended series
    val detrended = data.zip(trend).map { case (d, t) => d - t }
    
    // Calculate seasonal component
    val seasonalMap = MutableMap[Int, ArrayBuffer[Double]]()
    for (i <- detrended.indices) {
      val seasonIndex = i % period
      seasonalMap.getOrElseUpdate(seasonIndex, ArrayBuffer.empty) += detrended(i)
    }
    
    val seasonalFactors = (0 until period).map { i =>
      val values = seasonalMap.getOrElse(i, ArrayBuffer(0.0))
      values.sum / values.length
    }
    
    // Normalize seasonal factors to sum to zero
    val seasonalMean = seasonalFactors.sum / seasonalFactors.length
    val normalizedSeasonal = seasonalFactors.map(_ - seasonalMean)
    
    val seasonal = (0 until data.length).map(i => normalizedSeasonal(i % period))
    
    // Calculate residual
    val residual = (0 until data.length).map { i =>
      data(i) - trend(i) - seasonal(i)
    }
    
    TimeSeriesComponents(trend.toSeq, seasonal, residual, period)
  }
  
  /**
   * Auto-Regressive forecast (AR model)
   */
  def autoRegressive(data: Seq[Double], lags: Int = 3): ForecastResult = {
    if (data.length < lags + 1) {
      return ForecastResult(
        forecast = data.lastOption.getOrElse(0.0),
        lower95 = 0.0,
        upper95 = 0.0,
        trend = 0.0,
        seasonal = 1.0,
        method = "ar_insufficient_data"
      )
    }
    
    // Simple AR model using least squares
    val X = ArrayBuffer[Seq[Double]]()
    val y = ArrayBuffer[Double]()
    
    for (i <- lags until data.length) {
      X += data.slice(i - lags, i).reverse
      y += data(i)
    }
    
    // Calculate coefficients (simplified approach)
    val coefficients = (0 until lags).map { lag =>
      val corr = calculateCorrelation(
        data.drop(lags).dropRight(lag),
        data.drop(lags + lag)
      )
      corr / (lags + 1)
    }
    
    // Forecast next value
    val recentData = data.takeRight(lags).reverse
    val forecast = coefficients.zip(recentData).map { case (c, d) => c * d }.sum
    
    // Estimate error
    val predictions = (lags until data.length).map { i =>
      val recent = data.slice(i - lags, i).reverse
      coefficients.zip(recent).map { case (c, d) => c * d }.sum
    }
    
    val errors = predictions.zip(data.drop(lags)).map { case (pred, actual) => abs(actual - pred) }
    val mae = errors.sum / errors.length
    
    ForecastResult(
      forecast = max(0, forecast),
      lower95 = max(0, forecast - 1.96 * mae),
      upper95 = forecast + 1.96 * mae,
      trend = coefficients.headOption.getOrElse(0.0),
      seasonal = 1.0,
      method = "auto_regressive"
    )
  }
  
  /**
   * Calculate correlation between two sequences
   */
  private def calculateCorrelation(x: Seq[Double], y: Seq[Double]): Double = {
    if (x.length != y.length || x.isEmpty) return 0.0
    
    val meanX = x.sum / x.length
    val meanY = y.sum / y.length
    
    val covariance = x.zip(y).map { case (xi, yi) =>
      (xi - meanX) * (yi - meanY)
    }.sum
    
    val stdX = sqrt(x.map(xi => pow(xi - meanX, 2)).sum)
    val stdY = sqrt(y.map(yi => pow(yi - meanY, 2)).sum)
    
    if (stdX == 0 || stdY == 0) 0.0
    else covariance / (stdX * stdY)
  }
  
  /**
   * Ensemble forecast combining multiple methods
   */
  def ensembleForecast(
    data: Seq[Double],
    seasonLength: Int = 7
  ): ForecastResult = {
    if (data.length < 10) {
      return ForecastResult(
        forecast = data.lastOption.getOrElse(0.0),
        lower95 = 0.0,
        upper95 = 0.0,
        trend = 0.0,
        seasonal = 1.0,
        method = "insufficient_data"
      )
    }
    
    // Get forecasts from multiple methods
    val hwForecast = holtWinters(data, seasonLength)
    val arForecast = autoRegressive(data, min(5, data.length / 3))
    
    val esSmoothed = exponentialSmoothing(data)
    val esForecast = esSmoothed.lastOption.getOrElse(data.last)
    
    val maSmoothed = movingAverage(data, min(7, data.length / 2))
    val maForecast = maSmoothed.lastOption.getOrElse(data.last)
    
    // Weighted ensemble (weights based on typical performance)
    val weights = Map(
      "hw" -> 0.4,
      "ar" -> 0.3,
      "es" -> 0.2,
      "ma" -> 0.1
    )
    
    val ensembleForecast = 
      hwForecast.forecast * weights("hw") +
      arForecast.forecast * weights("ar") +
      esForecast * weights("es") +
      maForecast * weights("ma")
    
    val lower95 = 
      hwForecast.lower95 * weights("hw") +
      arForecast.lower95 * weights("ar") +
      (esForecast - 1.96 * data.map(d => abs(d - esForecast)).sum / data.length) * weights("es") +
      (maForecast - 1.96 * data.map(d => abs(d - maForecast)).sum / data.length) * weights("ma")
    
    val upper95 = 
      hwForecast.upper95 * weights("hw") +
      arForecast.upper95 * weights("ar") +
      (esForecast + 1.96 * data.map(d => abs(d - esForecast)).sum / data.length) * weights("es") +
      (maForecast + 1.96 * data.map(d => abs(d - maForecast)).sum / data.length) * weights("ma")
    
    ForecastResult(
      forecast = max(0, ensembleForecast),
      lower95 = max(0, lower95),
      upper95 = upper95,
      trend = hwForecast.trend,
      seasonal = hwForecast.seasonal,
      method = "ensemble"
    )
  }
  
  /**
   * Generate synthetic demand data for testing
   */
  def generateSyntheticDemand(
    length: Int,
    baseDemand: Double,
    trend: Double = 0.0,
    seasonalAmplitude: Double = 0.0,
    seasonalPeriod: Int = 7,
    noiseLevel: Double = 0.1
  ): Seq[Double] = {
    val random = new Random()
    
    (0 until length).map { t =>
      val trendComponent = trend * t
      val seasonalComponent = if (seasonalAmplitude > 0) {
        seasonalAmplitude * sin(2 * Pi * t / seasonalPeriod)
      } else 0.0
      val noise = random.nextGaussian() * noiseLevel * baseDemand
      
      max(0, baseDemand + trendComponent + seasonalComponent + noise)
    }
  }
}

// Optimizer using forecasted demand
class NewsvendorOptimizer(forecaster: DemandForecaster) {
  
  def optimizeWithForecast(
    historicalDemand: Seq[Double],
    params: Parameters,
    seasonLength: Int = 7
  ): (Int, ForecastResult) = {
    
    // Get demand forecast
    val forecast = forecaster.ensembleForecast(historicalDemand, seasonLength)
    
    // Calculate optimal Q using forecasted mean and confidence interval
    val cu = params.price - params.cost
    val co = params.cost + params.disposal - params.salvage
    val criticalFractile = cu / (cu + co)
    
    // Use forecast mean and adjust for uncertainty
    val forecastMean = forecast.forecast
    val forecastStd = (forecast.upper95 - forecast.lower95) / (2 * 1.96)
    
    // Optimal Q using inverse normal CDF approximation
    val z = inverseNormalCDF(criticalFractile)
    val optimalQ = (forecastMean + z * forecastStd).toInt
    
    (max(0, optimalQ), forecast)
  }
  
  private def inverseNormalCDF(p: Double): Double = {
    // Beasley-Springer-Moro algorithm
    val a = Array(-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00)
    
    val b = Array(-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01)
    
    val c = Array(-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00)
    
    val d = Array(7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00)
    
    val pLow = 0.02425
    val pHigh = 1 - pLow
    
    if (p < pLow) {
      val q = sqrt(-2 * log(p))
      (((((c(0)*q+c(1))*q+c(2))*q+c(3))*q+c(4))*q+c(5)) /
        ((((d(0)*q+d(1))*q+d(2))*q+d(3))*q+1)
    } else if (p <= pHigh) {
      val q = p - 0.5
      val r = q * q
      (((((a(0)*r+a(1))*r+a(2))*r+a(3))*r+a(4))*r+a(5))*q /
        (((((b(0)*r+b(1))*r+b(2))*r+b(3))*r+b(4))*r+1)
    } else {
      val q = sqrt(-2 * log(1 - p))
      -(((((c(0)*q+c(1))*q+c(2))*q+c(3))*q+c(4))*q+c(5)) /
        ((((d(0)*q+d(1))*q+d(2))*q+d(3))*q+1)
    }
  }
}

// Main application
object NewsvendorForecasting extends App {
  
  println("╔══════════════════════════════════════════════════════════╗")
  println("║                                                          ║")
  println("║     NEWSVENDOR FORECASTING ENGINE - SCALA                ║")
  println("║                                                          ║")
  println("║     Advanced demand forecasting with time series         ║")
  println("║                                                          ║")
  println("╚══════════════════════════════════════════════════════════╝")
  println()
  
  val forecaster = new DemandForecaster()
  val optimizer = new NewsvendorOptimizer(forecaster)
  
  // Generate synthetic demand data with trend and seasonality
  val syntheticDemand = forecaster.generateSyntheticDemand(
    length = 60,
    baseDemand = 100.0,
    trend = 0.5,
    seasonalAmplitude = 15.0,
    seasonalPeriod = 7,
    noiseLevel = 0.15
  )
  
  println("Generated 60 days of synthetic demand data")
  println(s"Mean: ${syntheticDemand.sum / syntheticDemand.length}")
  println(s"Min: ${syntheticDemand.min}, Max: ${syntheticDemand.max}")
  println()
  
  // Test different forecasting methods
  println("Testing Forecasting Methods:")
  println("-" * 60)
  
  val hwForecast = forecaster.holtWinters(syntheticDemand, 7)
  println(s"Holt-Winters: ${hwForecast.forecast.toInt} " +
          s"[${hwForecast.lower95.toInt}, ${hwForecast.upper95.toInt}]")
  
  val arForecast = forecaster.autoRegressive(syntheticDemand, 5)
  println(s"Auto-Regressive: ${arForecast.forecast.toInt} " +
          s"[${arForecast.lower95.toInt}, ${arForecast.upper95.toInt}]")
  
  val ensembleForecast = forecaster.ensembleForecast(syntheticDemand, 7)
  println(s"Ensemble: ${ensembleForecast.forecast.toInt} " +
          s"[${ensembleForecast.lower95.toInt}, ${ensembleForecast.upper95.toInt}]")
  
  println()
  
  // Optimize with forecast
  val params = Parameters(
    price = 5.0,
    cost = 2.0,
    disposal = 0.5,
    salvage = 0.0,
    demandMean = 100.0,
    demandStd = 20.0
  )
  
  val (optimalQ, forecast) = optimizer.optimizeWithForecast(syntheticDemand, params)
  
  println("Optimization Results:")
  println("-" * 60)
  println(s"Forecasted Demand: ${forecast.forecast.toInt}")
  println(s"Optimal Order Quantity: $optimalQ")
  println(s"Forecast Method: ${forecast.method}")
  println(s"Trend: ${forecast.trend}")
  println(s"Seasonal Factor: ${forecast.seasonal}")
  
  println()
  println("Forecasting engine ready for production use.")
}