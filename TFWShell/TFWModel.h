#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../SecondFundamentalForm/SecondFundamentalFormDiscretization.h"
#include "../SecondFundamentalForm/MidedgeAverageFormulation.h"
#include "../SecondFundamentalForm/MidedgeAngleSinFormulation.h"
#include "../SecondFundamentalForm/MidedgeAngleTanFormulation.h"
#include "CommonFunctions.h"

#include "TFWSetup.h"
#include "TFWState.h"

using namespace TFW;

class TFWModel
{
public:
	void initialization(const TFWSetup setup, const TFWState state, std::map<int, double> *clampedDOFs, std::string filePath, bool isUsePosHess = false, bool isParallel = false);

	void setProjM(std::map<int, double> *clampedDOFs);

	void convertCurState2Variables(const TFWState curState, Eigen::VectorXd &x);
	void convertVariables2CurState(const Eigen::VectorXd x, TFWState &curState);

	double value(const Eigen::VectorXd &x);
	double stretchingValue(const Eigen::VectorXd &x);
	double bendingValue(const Eigen::VectorXd &x);

	void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad);
	void hessian(const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& hessian);     
	Eigen::VectorXd getProjectedGradient(const Eigen::VectorXd &x);

	void testValueAndGradient(const Eigen::VectorXd &x);
	void testGradientAndHessian(const Eigen::VectorXd& x);

    
	Eigen::VectorXd getFullDir(const Eigen::VectorXd &dir) { return _projM.transpose() * dir; }

	int nFreeAmp() {return _freeAmp; }
	int nFreePhi() {return _freePhi; }
	int nFreeDphi() { return _freeDphi; }



	Eigen::VectorXd fullGradient(const Eigen::VectorXd &grad) {return _projM.transpose() * grad;}

	bool isC2() { return true; }

	double getMaxStep(const Eigen::VectorXd& x, const Eigen::VectorXd& dir, double step) 
	{
		return 1e15;
	}

public:
	TFWSetup _setup;
	TFWState _state;
	//Eigen::MatrixXd _prevPos;
	Eigen::SparseMatrix<double> _projM;
	Eigen::MatrixXd _reducedBasis;
	std::map<int, double> _clampedDOFs;

	bool _isUseD2phi;
	bool _isUsePosHess;

	bool _isParallel;

	int _freeAmp;
	int _freePhi;
	int _freeDphi;

	double _reg;
};
