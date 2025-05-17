package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
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

// Matrix-vector product
func matVecMul(X [][]float64, w []float64) []float64 {
	m := len(X)
	h := make([]float64, m)
	for i := 0; i < m; i++ {
		h[i] = sigmoid(dot(X[i], w))
	}
	return h
}

// Compute cost (log-loss)
func computeCost(X [][]float64, y []float64, w []float64) float64 {
	m := float64(len(y))
	h := matVecMul(X, w)
	epsilon := 1e-5
	cost := 0.0
	for i := range y {
		cost += -y[i]*math.Log(h[i]+epsilon) - (1-y[i])*math.Log(1-h[i]+epsilon)
	}
	return cost / m
}

// Transpose of a matrix
func transpose(X [][]float64) [][]float64 {
	rows, cols := len(X), len(X[0])
	T := make([][]float64, cols)
	for i := range T {
		T[i] = make([]float64, rows)
		for j := range T[i] {
			T[i][j] = X[j][i]
		}
	}
	return T
}

// Gradient descent
func gradientDescent(X [][]float64, y []float64, weights []float64, lr float64, epochs int) ([]float64, []float64) {
	m := float64(len(y))
	costHistory := make([]float64, 0, epochs)

	for i := 0; i < epochs; i++ {
		h := matVecMul(X, weights)
		gradient := make([]float64, len(weights))
		for j := range weights {
			sum := 0.0
			for k := range X {
				sum += (h[k] - y[k]) * X[k][j]
			}
			gradient[j] = sum / m
			weights[j] -= lr * gradient[j]
		}
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

	// Agregar bias manualmente
	for i := range X {
		X[i] = append([]float64{1.0}, X[i]...) // prepend bias
	}

	// Inicializar pesos en cero (una por cada feature + bias)
	weights := make([]float64, len(X[0]))

	return X, Y, weights, nil
}

func main() {
	// numSamples := 100
	// X, y := generateData(numSamples, 3)
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
	//fmt.Printf("Final Weights: %.4f %.4f %.4f\n", weights[0], weights[1], weights[2])
	fmt.Printf("Final Cost: %.4f\n", costHistory[len(costHistory)-1])
}
