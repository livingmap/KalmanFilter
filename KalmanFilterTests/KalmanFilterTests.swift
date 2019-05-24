//
//  KalmanFilterTests.swift
//  KalmanFilterTests
//
//  Created by Oleksii on 17/08/16.
//  Copyright © 2016 Oleksii Dykan. All rights reserved.
//

import XCTest
@testable import KalmanFilter

class KalmanFilterTests: XCTestCase {
    
    let GENERAL_WIFI_ACCURACY_METERS = 2.0
    let STRIDE_LENGTH_VARIANCE_METERS = 0.1
    let MEAN_STRIDE_LENGTH_METERS = 0.762
    
    var measurementMatrix: Matrix? = nil
    var measurementNoise: Matrix? = nil
    var processNoise: Matrix? = nil
    var stateTransitionMatrix: Matrix? = nil
    var controlMatrix: Matrix? = nil
    
    override func setUp() {
        
        self.measurementMatrix = Matrix(identityOfSize: 2)
        self.stateTransitionMatrix = Matrix(identityOfSize: 2)
        self.controlMatrix = Matrix(identityOfSize: 2)
        
        self.measurementNoise = Matrix([
            [pow(GENERAL_WIFI_ACCURACY_METERS, 2.0), 0.0],
            [0.0, pow(GENERAL_WIFI_ACCURACY_METERS, 2.0)]
        ])
        
        self.processNoise = Matrix([
            [STRIDE_LENGTH_VARIANCE_METERS, 0.0],
            [0.0, STRIDE_LENGTH_VARIANCE_METERS]
        ])
    }
    
    func kalmanFilterCorrect(kalmanFilter: KalmanFilter<Matrix>, measurement: Matrix) -> KalmanFilter<Matrix> {
        // introduce noise
        let kalmanFilter = kalmanFilter.predict(
            stateTransitionModel: stateTransitionMatrix!,
            controlInputModel: controlMatrix!,
            controlVector: Matrix(vector: [0, 0]),
            covarianceOfProcessNoise: processNoise!
        )
        return kalmanFilter.update(
            measurement: measurement,
            observationModel: measurementMatrix!,
            covarienceOfObservationNoise: measurementNoise!
        )
    }
    
    func kalmanFilterPredict(kalmanFilter: KalmanFilter<Matrix>, controlVector: Matrix) -> KalmanFilter<Matrix> {
        return kalmanFilter.predict(
            stateTransitionModel: stateTransitionMatrix!,
            controlInputModel: controlMatrix!,
            controlVector: controlVector,
            covarianceOfProcessNoise: processNoise!
        )
    }
    
    func testLivingMap() {
        let initialStateEstimate = Matrix(vector: [530369.1952072862, 181399.91357055644])
        let finalStateEstimate = Matrix(vector: [530375.1952072862, 181405.91357055644])
        let strideVector = Matrix(vector: [MEAN_STRIDE_LENGTH_METERS / 2, MEAN_STRIDE_LENGTH_METERS / 2])
        
        var kalmanFilter = KalmanFilter(stateEstimatePrior: initialStateEstimate, errorCovariancePrior: measurementNoise!)
        kalmanFilter = kalmanFilterPredict(kalmanFilter: kalmanFilter, controlVector: strideVector)
        kalmanFilter = kalmanFilterPredict(kalmanFilter: kalmanFilter, controlVector: strideVector)
        kalmanFilter = kalmanFilterPredict(kalmanFilter: kalmanFilter, controlVector: strideVector)
        kalmanFilter = kalmanFilterCorrect(kalmanFilter: kalmanFilter, measurement: finalStateEstimate)
        
        let accuracy = 0.0000000001
        XCTAssertEqual(kalmanFilter.stateEstimatePrior.grid[0], 530372.8823501434, accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.stateEstimatePrior.grid[1], 181403.6007134136, accuracy: accuracy)
    }
    
    func testKalmanFilter2D() {
        let measurements = [1.0, 2.0, 3.0]
        let accuracy = 0.00001
        
        let x = Matrix(vector: [0, 0])
        let P = Matrix(grid: [1000, 0, 0, 1000], rows: 2, columns: 2)
        let B = Matrix(identityOfSize: 2)
        let u = Matrix(vector: [0, 0])
        let F = Matrix(grid: [1, 1, 0, 1], rows: 2, columns: 2)
        let H = Matrix(grid: [1, 0], rows: 1, columns: 2)
        let R = Matrix(grid: [1], rows: 1, columns: 1)
        let Q = Matrix(rows: 2, columns: 2)
        
        var kalmanFilter = KalmanFilter(stateEstimatePrior: x, errorCovariancePrior: P)
        
        for measurement in measurements {
            let z = Matrix(grid: [measurement], rows: 1, columns: 1)
            kalmanFilter = kalmanFilter.update(measurement: z, observationModel: H, covarienceOfObservationNoise: R)
            kalmanFilter = kalmanFilter.predict(stateTransitionModel: F, controlInputModel: B, controlVector: u, covarianceOfProcessNoise: Q)
        }
        
        let resultX = Matrix(vector: [3.9996664447958645, 0.9999998335552873])
        let resultP = Matrix(grid: [2.3318904241194827, 0.9991676099921091, 0.9991676099921067, 0.49950058263974184], rows: 2, columns: 2)
        
        XCTAssertEqual(kalmanFilter.stateEstimatePrior[0, 0], resultX[0, 0], accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.stateEstimatePrior[1, 0], resultX[1, 0], accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.errorCovariancePrior[0, 0], resultP[0, 0], accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.errorCovariancePrior[0, 1], resultP[0, 1], accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.errorCovariancePrior[1, 0], resultP[1, 0], accuracy: accuracy)
        XCTAssertEqual(kalmanFilter.errorCovariancePrior[1, 1], resultP[1, 1], accuracy: accuracy)
    }
    
}
