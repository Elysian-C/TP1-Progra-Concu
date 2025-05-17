package main

import (
	"fmt"
	"encoding/csv"
	"log"
	"math"
	"sync"
	"os"
	"io"
	"strconv"
	"time"
)

// Sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Dot product between two vectors
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		log.Fatal("vector sizes do not match")
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Matrix-vector product (concurrent)
func matVecMul(X [][]float64, w []float64) []float64 {
	m := len(X)
	h := make([]float64, m)
	var wg sync.WaitGroup

	for i := 0; i < m; i++ {
		wg.Add(1)
		go func(i int) {
			h[i] = sigmoid(dot(X[i], w))
			wg.Done()
		}(i)
	}
	wg.Wait()
	return h
}

// Compute cost (log-loss)
func computeCost(X [][]float64, y []float64, w []float64) float64 {
	m := float64(len(y))
	h := matVecMul(X, w)
	epsilon := 1e-5
	cost := 0.0
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := range y {
		wg.Add(1)
		go func(i int) {
			localCost := -y[i]*math.Log(h[i]+epsilon) - (1-y[i])*math.Log(1-h[i]+epsilon)
			mu.Lock()
			cost += localCost
			mu.Unlock()
			wg.Done()
		}(i)
	}
	wg.Wait()
	return cost / m
}

// Gradient descent (with concurrency)
func gradientDescent(X [][]float64, y []float64, weights []float64, lr float64, epochs int) ([]float64, []float64) {
	m := float64(len(y))
	costHistory := make([]float64, 0, epochs)
	
	for i := 0; i < epochs; i++ {
		h := matVecMul(X, weights)
		gradient := make([]float64, len(weights))
		var wg sync.WaitGroup
		var mu sync.Mutex

		for j := range weights {
			wg.Add(1)
			go func(j int) {
				sum := 0.0
				for k := range X {
					sum += (h[k] - y[k]) * X[k][j]
				}
				mu.Lock()
				gradient[j] = sum / m
				weights[j] -= lr * gradient[j]
				mu.Unlock()
				wg.Done()
			}(j)
		}
		wg.Wait()

		cost := computeCost(X, y, weights)
		costHistory = append(costHistory, cost)
	}
	return weights, costHistory
}

func loadData(csvPath string) ([][]float64, []float64, []float64, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error abriendo archivo CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	_, err = reader.Read()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error leyendo encabezado: %v", err)
	}

	var X [][]float64
	var Y []float64
	id := 0;
	for {
		id ++;
		if id>10000 { break ;}
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("error leyendo línea del CSV: %v", err)
		}

		n := len(record)
		sample := make([]float64, n-1)
		for i := 0; i < n-1; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("error parseando valor X: %v", err)
			}
			sample[i] = val
		}
		X = append(X, sample)

		label, err := strconv.ParseFloat(record[n-1], 64)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("error parseando valor Y: %v", err)
		}
		Y = append(Y, label)
	}

	if len(X) == 0 {
		return nil, nil, nil, fmt.Errorf("el archivo está vacío o solo contiene encabezado")
	}

	for i := range X {
		X[i] = append([]float64{1.0}, X[i]...) // prepend bias
	}
	weights := make([]float64, len(X[0]))

	return X, Y, weights, nil
}

func main() {
	
	X, y, weights,er := loadData("heart_1M.csv")
	if er != nil {
		fmt.Print(er)
	}
	learningRate := 0.1
	epochs := 1000
	start := time.Now()
	weights, costHistory := gradientDescent(X, y, weights, learningRate, epochs)
	elapsed := time.Since(start)
	fmt.Printf("Tiempo total de ejecución: %s\n", elapsed)
	//fmt.Printf("Final Weights: %.4f %.4f %.4f %.4f\n", weights[0], weights[1], weights[2], weights[3])
	fmt.Printf("Final Cost: %.4f\n", costHistory[len(costHistory)-1])
	
}

