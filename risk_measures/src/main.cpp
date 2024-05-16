#include <iostream>
#include <vector>
#include "matrix.hpp"

class RiskMeasures {
private:
    Matrix returns;
    Matrix weights;
public:
    RiskMeasures(Matrix returns, Matrix weights) : returns(returns), weights(weights) {
        checkWeights();
    }

    void covariance() {
    }

    Matrix productReturns() {
        Matrix product_returns;
        for (int i = 0; i < returns.getCols(); i++) {
            for (int j = 0; j < returns.getCols(); j++) {
                returns(ALL, i) * returns(ALL, j);
            }
        }
    }

private:
    void checkWeights() {
        if (weights.getRows() != 1 && weights.getCols() != 3) {
            throw std::invalid_argument("Invalid weights matrix dimensions.");
        }
        double sum = weights.sum();
        if (sum != 1) {
            throw std::invalid_argument("Weights do not sum to 1.");
        }
    }
};

int main() {
    Matrix matrix1 = Matrix({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    Matrix matrix2 = Matrix(std::vector<std::vector<double>>{{7.0, 8.0, 9.0}});
    Matrix matrix3 = Matrix({std::string{"[[7],[8],[9]]"}});
    Matrix x = matrix2 % matrix1 % matrix3;
    x = x / 2;
    matrix1 = ~matrix1;
    std::string str = matrix1.toString();
    std::cout << str << std::endl;

    Matrix y = matrix1(1, 2);
    Matrix z = matrix1(1, ALL);
    Matrix a = matrix1(ALL, 2);
    // std::cout << y << std::endl;
    // std::cout << z << std::endl;
    // std::cout << a << std::endl;
    // std::cout << matrix1 << std::endl;

    Matrix b;

    Matrix c = Matrix(std::vector<std::vector<double>>{{1.0, 2.0}});
    std::vector<double> v = {1.0, 2.0};
    // b.appendRow(c);
    std::cout << c << std::endl;
    b.append(c);
    b.append(c);
    std::cout << b << std::endl;
    // b.append(v);
    // std::cout << b << std::endl;


    return 0;

}