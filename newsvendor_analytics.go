package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

/*
Newsvendor Analytics Engine - Go Implementation

High-performance analytics service for newsvendor optimization
Features:
- Concurrent Monte Carlo simulation
- Statistical distribution analysis
- Profit optimization with goroutines
- RESTful API with JSON responses
*/

// Parameters represents newsvendor problem parameters
type Parameters struct {
	Price         float64 `json:"price"`
	Cost          float64 `json:"cost"`
	Disposal      float64 `json:"disposal"`
	Salvage       float64 `json:"salvage"`
	DemandMean    float64 `json:"demandMean"`
	DemandStd     float64 `json:"demandStd"`
	Distribution  string  `json:"distribution"`
	PoissonLambda float64 `json:"poissonLambda"`
	LeadTime      int     `json:"leadTime"`
}

// OptimizationResult contains optimization results
type OptimizationResult struct {
	OptimalQ         int     `json:"optimalQ"`
	CriticalFractile float64 `json:"criticalFractile"`
	ExpectedProfit   float64 `json:"expectedProfit"`
	UnderageCost     float64 `json:"underageCost"`
	OverageCost      float64 `json:"overageCost"`
	ComputeTime      float64 `json:"computeTimeMs"`
}

// SimulationResult contains Monte Carlo simulation results
type SimulationResult struct {
	Iterations     int                    `json:"iterations"`
	AvgProfit      float64                `json:"avgProfit"`
	StdDev         float64                `json:"stdDev"`
	Min            float64                `json:"min"`
	Max            float64                `json:"max"`
	Percentiles    map[string]float64     `json:"percentiles"`
	ConfidenceInt  map[string]float64     `json:"confidenceInterval"`
	Scenarios      []Scenario             `json:"scenarios"`
	ComputeTime    float64                `json:"computeTimeMs"`
}

// Scenario represents a single simulation run
type Scenario struct {
	Iteration int     `json:"iteration"`
	Demand    int     `json:"demand"`
	Sales     int     `json:"sales"`
	Leftover  int     `json:"leftover"`
	Profit    float64 `json:"profit"`
}

// ProfitCurvePoint represents a point on the profit curve
type ProfitCurvePoint struct {
	Quantity       int     `json:"quantity"`
	ExpectedProfit float64 `json:"expectedProfit"`
}

// Statistics helper struct
type Statistics struct {
	data []float64
	mu   sync.Mutex
}

// NewStatistics creates a new Statistics instance
func NewStatistics() *Statistics {
	return &Statistics{
		data: make([]float64, 0),
	}
}

// Add adds a value to the statistics
func (s *Statistics) Add(value float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = append(s.data, value)
}

// Mean calculates the mean
func (s *Statistics) Mean() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range s.data {
		sum += v
	}
	return sum / float64(len(s.data))
}

// StdDev calculates the standard deviation
func (s *Statistics) StdDev() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.data) == 0 {
		return 0
	}
	mean := s.Mean()
	variance := 0.0
	for _, v := range s.data {
		variance += math.Pow(v-mean, 2)
	}
	return math.Sqrt(variance / float64(len(s.data)))
}

// Percentile calculates the percentile
func (s *Statistics) Percentile(p float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.data) == 0 {
		return 0
	}
	// Sort data
	sorted := make([]float64, len(s.data))
	copy(sorted, s.data)
	sortFloat64(sorted)
	
	index := int(math.Ceil(float64(len(sorted)) * p))
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

// Min returns minimum value
func (s *Statistics) Min() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.data) == 0 {
		return 0
	}
	min := s.data[0]
	for _, v := range s.data {
		if v < min {
			min = v
		}
	}
	return min
}

// Max returns maximum value
func (s *Statistics) Max() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.data) == 0 {
		return 0
	}
	max := s.data[0]
	for _, v := range s.data {
		if v > max {
			max = v
		}
	}
	return max
}

// sortFloat64 sorts a float64 slice (simple bubble sort for small arrays)
func sortFloat64(arr []float64) {
	n := len(arr)
	for i := 0; i < n; i++ {
		for j := 0; j < n-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

// NormalCDF computes the cumulative distribution function for normal distribution
func NormalCDF(x, mean, std float64) float64 {
	z := (x - mean) / std
	t := 1.0 / (1.0 + 0.2316419*math.Abs(z))
	d := 0.3989423 * math.Exp(-z*z/2)
	p := d * t * (0.3193815 + t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))))
	if z > 0 {
		return 1 - p
	}
	return p
}

// NormalPDF computes the probability density function for normal distribution
func NormalPDF(x, mean, std float64) float64 {
	return (1.0 / (std * math.Sqrt(2*math.Pi))) *
		math.Exp(-0.5*math.Pow((x-mean)/std, 2))
}

// PoissonPMF computes the probability mass function for Poisson distribution
func PoissonPMF(k int, lambda float64) float64 {
	if k < 0 || k > 500 {
		return 0
	}
	result := math.Exp(-lambda)
	for i := 1; i <= k; i++ {
		result *= lambda / float64(i)
	}
	return result
}

// InverseNormalCDF computes the inverse CDF (quantile function)
func InverseNormalCDF(p, mean, std float64) float64 {
	if p <= 0 || p >= 1 {
		return mean
	}
	
	// Beasley-Springer-Moro algorithm approximation
	a := []float64{
		-3.969683028665376e+01, 2.209460984245205e+02,
		-2.759285104469687e+02, 1.383577518672690e+02,
		-3.066479806614716e+01, 2.506628277459239e+00,
	}
	
	b := []float64{
		-5.447609879822406e+01, 1.615858368580409e+02,
		-1.556989798598866e+02, 6.680131188771972e+01,
		-1.328068155288572e+01,
	}
	
	c := []float64{
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00, 2.938163982698783e+00,
	}
	
	d := []float64{
		7.784695709041462e-03, 3.224671290700398e-01,
		2.445134137142996e+00, 3.754408661907416e+00,
	}
	
	pLow := 0.02425
	pHigh := 1 - pLow
	
	var q, r, x float64
	
	if p < pLow {
		q = math.Sqrt(-2 * math.Log(p))
		x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	} else if p <= pHigh {
		q = p - 0.5
		r = q * q
		x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	} else {
		q = math.Sqrt(-2 * math.Log(1-p))
		x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
	
	return mean + std*x
}

// GenerateDemand generates a random demand sample
func GenerateDemand(params Parameters) int {
	if params.Distribution == "normal" {
		u1 := rand.Float64()
		u2 := rand.Float64()
		z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		demand := params.DemandMean + z*params.DemandStd
		if demand < 0 {
			return 0
		}
		return int(math.Round(demand))
	} else if params.Distribution == "poisson" {
		L := math.Exp(-params.PoissonLambda)
		k := 0
		p := 1.0
		for p > L && k < 1000 {
			k++
			p *= rand.Float64()
		}
		return k - 1
	}
	return int(params.DemandMean)
}

// CalculateOptimalQ computes the optimal order quantity
func CalculateOptimalQ(params Parameters) OptimizationResult {
	startTime := time.Now()
	
	cu := params.Price - params.Cost
	co := params.Cost + params.Disposal - params.Salvage
	criticalFractile := cu / (cu + co)
	
	var optimalQ int
	
	if params.Distribution == "normal" {
		optimalQ = int(math.Round(InverseNormalCDF(criticalFractile, params.DemandMean, params.DemandStd)))
	} else if params.Distribution == "poisson" {
		// Binary search for Poisson
		low := 0
		high := int(params.PoissonLambda * 3)
		
		for low < high {
			mid := (low + high) / 2
			cdf := 0.0
			for k := 0; k <= mid; k++ {
				cdf += PoissonPMF(k, params.PoissonLambda)
			}
			
			if cdf < criticalFractile {
				low = mid + 1
			} else {
				high = mid
			}
		}
		optimalQ = low
	}
	
	if optimalQ < 0 {
		optimalQ = 0
	}
	
	expectedProfit := CalculateExpectedProfit(optimalQ, params)
	computeTime := time.Since(startTime).Seconds() * 1000
	
	return OptimizationResult{
		OptimalQ:         optimalQ,
		CriticalFractile: criticalFractile,
		ExpectedProfit:   expectedProfit,
		UnderageCost:     cu,
		OverageCost:      co,
		ComputeTime:      computeTime,
	}
}

// CalculateExpectedProfit computes expected profit for given Q
func CalculateExpectedProfit(Q int, params Parameters) float64 {
	expectedSales := 0.0
	expectedLeftover := 0.0
	maxDemand := int(params.DemandMean * 3)
	
	if params.Distribution == "poisson" {
		maxDemand = int(params.PoissonLambda * 3)
	}
	
	for d := 0; d <= maxDemand; d++ {
		var prob float64
		if params.Distribution == "normal" {
			prob = NormalPDF(float64(d), params.DemandMean, params.DemandStd)
		} else {
			prob = PoissonPMF(d, params.PoissonLambda)
		}
		
		sales := math.Min(float64(d), float64(Q))
		leftover := math.Max(0, float64(Q-d))
		
		expectedSales += sales * prob
		expectedLeftover += leftover * prob
	}
	
	revenue := params.Price * expectedSales
	productionCost := params.Cost * float64(Q)
	disposalCost := params.Disposal * expectedLeftover
	salvageRevenue := params.Salvage * expectedLeftover
	
	return revenue - productionCost - disposalCost + salvageRevenue
}

// RunMonteCarloSimulation performs concurrent Monte Carlo simulation
func RunMonteCarloSimulation(Q int, params Parameters, iterations int) SimulationResult {
	startTime := time.Now()
	
	stats := NewStatistics()
	scenarios := make([]Scenario, 0, 100)
	var scenarioMu sync.Mutex
	
	// Use goroutines for parallel simulation
	numWorkers := 8
	iterationsPerWorker := iterations / numWorkers
	var wg sync.WaitGroup
	
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			// Local random source for thread safety
			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)))
			
			for i := 0; i < iterationsPerWorker; i++ {
				// Generate demand with local random source
				var demand int
				if params.Distribution == "normal" {
					u1 := localRand.Float64()
					u2 := localRand.Float64()
					z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
					demandFloat := params.DemandMean + z*params.DemandStd
					if demandFloat < 0 {
						demand = 0
					} else {
						demand = int(math.Round(demandFloat))
					}
				} else if params.Distribution == "poisson" {
					L := math.Exp(-params.PoissonLambda)
					k := 0
					p := 1.0
					for p > L && k < 1000 {
						k++
						p *= localRand.Float64()
					}
					demand = k - 1
				}
				
				sales := int(math.Min(float64(demand), float64(Q)))
				leftover := int(math.Max(0, float64(Q-demand)))
				profit := params.Price*float64(sales) -
					params.Cost*float64(Q) -
					params.Disposal*float64(leftover) +
					params.Salvage*float64(leftover)
				
				stats.Add(profit)
				
				// Store first 100 scenarios
				scenarioMu.Lock()
				if len(scenarios) < 100 {
					scenarios = append(scenarios, Scenario{
						Iteration: workerID*iterationsPerWorker + i + 1,
						Demand:    demand,
						Sales:     sales,
						Leftover:  leftover,
						Profit:    math.Round(profit*100) / 100,
					})
				}
				scenarioMu.Unlock()
			}
		}(w)
	}
	
	wg.Wait()
	
	mean := stats.Mean()
	stdDev := stats.StdDev()
	
	// Calculate confidence interval (95%)
	marginOfError := 1.96 * stdDev / math.Sqrt(float64(iterations))
	
	computeTime := time.Since(startTime).Seconds() * 1000
	
	return SimulationResult{
		Iterations: iterations,
		AvgProfit:  math.Round(mean*100) / 100,
		StdDev:     math.Round(stdDev*100) / 100,
		Min:        math.Round(stats.Min()*100) / 100,
		Max:        math.Round(stats.Max()*100) / 100,
		Percentiles: map[string]float64{
			"p5":  math.Round(stats.Percentile(0.05)*100) / 100,
			"p25": math.Round(stats.Percentile(0.25)*100) / 100,
			"p50": math.Round(stats.Percentile(0.50)*100) / 100,
			"p75": math.Round(stats.Percentile(0.75)*100) / 100,
			"p95": math.Round(stats.Percentile(0.95)*100) / 100,
		},
		ConfidenceInt: map[string]float64{
			"lower": math.Round((mean-marginOfError)*100) / 100,
			"upper": math.Round((mean+marginOfError)*100) / 100,
		},
		Scenarios:   scenarios,
		ComputeTime: computeTime,
	}
}

// HTTP Handlers

func optimizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var params Parameters
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	result := CalculateOptimalQ(params)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func simulateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var request struct {
		OrderQuantity int        `json:"orderQuantity"`
		Parameters    Parameters `json:"parameters"`
		Iterations    int        `json:"iterations"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	if request.Iterations == 0 {
		request.Iterations = 10000
	}
	
	result := RunMonteCarloSimulation(request.OrderQuantity, request.Parameters, request.Iterations)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func profitCurveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var request struct {
		QMin       int        `json:"qMin"`
		QMax       int        `json:"qMax"`
		Step       int        `json:"step"`
		Parameters Parameters `json:"parameters"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	if request.Step == 0 {
		request.Step = 2
	}
	
	curve := make([]ProfitCurvePoint, 0)
	for q := request.QMin; q <= request.QMax; q += request.Step {
		profit := CalculateExpectedProfit(q, request.Parameters)
		curve = append(curve, ProfitCurvePoint{
			Quantity:       q,
			ExpectedProfit: math.Round(profit*100) / 100,
		})
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(curve)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "healthy",
		"service":   "newsvendor-analytics",
		"timestamp": time.Now().Format(time.RFC3339),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())
	
	// Setup routes
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/optimize", optimizeHandler)
	http.HandleFunc("/simulate", simulateHandler)
	http.HandleFunc("/profit-curve", profitCurveHandler)
	
	// Enable CORS
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		http.DefaultServeMux.ServeHTTP(w, r)
	})
	
	port := ":8080"
	
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                          ║")
	fmt.Println("║     NEWSVENDOR ANALYTICS ENGINE - GO SERVICE             ║")
	fmt.Println("║                                                          ║")
	fmt.Printf("║     Server running on port %s                          ║\n", port)
	fmt.Println("║                                                          ║")
	fmt.Println("║     Endpoints:                                           ║")
	fmt.Println("║     - POST /optimize                                     ║")
	fmt.Println("║     - POST /simulate                                     ║")
	fmt.Println("║     - POST /profit-curve                                 ║")
	fmt.Println("║     - GET  /health                                       ║")
	fmt.Println("║                                                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	
	log.Fatal(http.ListenAndServe(port, handler))
}