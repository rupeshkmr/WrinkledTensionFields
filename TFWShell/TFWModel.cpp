#include <igl/writeOBJ.h>
#include <igl/null.h>
#include <memory>
#include <set>
#include "TFWShell.h"
#include "TFWModel.h"

void TFWModel::initialization(const TFWSetup setup, const TFWState state, std::map<int, double>* clampedDOFs, std::string filePath, bool isUsePosHess, bool isParallel)
{
	_setup = setup;
	//_prevPos = setup.initialPos;
	_state = state;
	setProjM(clampedDOFs);
	_isUsePosHess = isUsePosHess;
	_isParallel = isParallel;

	if (_isParallel)
		std::cout << "Use TBB for parallel computing the energy per face" << std::endl;
	else
		std::cout << "Sequential computing the energy per face" << std::endl;
}


void TFWModel::setProjM(std::map<int, double>* clampedDOFs)
{
	_freeAmp = 0;
	_freePhi = 0;
	_freeDphi = 0;
	MeshConnectivity mesh = _state.baseMesh;
	int nverts = _state.basePos.rows();
	int nedges = _state.baseMesh.nEdges();
	int nfaces = _state.baseMesh.nFaces();

	std::set<int> pureTensionEdges, pureTensionVertices;

	getPureTensionVertsEdges(_state.tensionFaces, _state.baseMesh.faces(), &pureTensionEdges, &pureTensionVertices);

	std::set<int> clampedAmps = pureTensionVertices;

	_clampedDOFs.clear();
	if (clampedDOFs)
		_clampedDOFs = *clampedDOFs;

	std::cout << "number of clamped amp before considering pure tension case: " << _clampedDOFs.size() << std::endl;

	for (auto& vid : clampedAmps) // clampedDOFs: 0-nverts: clampedAmp, > nverts: clamped phi or dphi
	{
		if (_clampedDOFs.find(vid) == _clampedDOFs.end())
		{
			_clampedDOFs[vid] = 0;
		}
	}

	std::cout << "number of clamped amp after considering pure tension case: " << _clampedDOFs.size() << std::endl;

	std::vector<Eigen::Triplet<double> > proj;
	for (auto& eid : pureTensionEdges)
	{
		_clampedDOFs[eid + nverts] = 0;
	}

	int constrainedDOFs = _clampedDOFs.size();

	int freeDOFs = nedges + nverts - constrainedDOFs;
	int row = 0;

	for (int i = 0; i < nedges + nverts; i++)
	{
		if (_clampedDOFs.find(i) != _clampedDOFs.end())
			continue;
		if (i < nverts)
			_freeAmp++;
		else
		{
			_freeDphi++;
		}
		proj.push_back(Eigen::Triplet<double>(row, i, 1.0));
		row++;
	}
	_projM.resize(freeDOFs, nedges + nverts);
	_projM.setFromTriplets(proj.begin(), proj.end());


	std::cout << "Free Amps: " << _freeAmp << std::endl;
	std::cout << "Free dphis: " << _freeDphi << std::endl;
}

void TFWModel::convertCurState2Variables(const TFWState curState, Eigen::VectorXd& x)
{
	int namp = curState.amplitude.size();
	int ndphi = curState.dphi.size();
	Eigen::VectorXd y(namp + ndphi);

	//y.segment(0,namp) = curState.amplitude * _setup.rescale;
	y.segment(0, namp) = curState.amplitude;
	y.segment(namp, ndphi) = curState.dphi;

	x = _projM * y;
}

void TFWModel::convertVariables2CurState(Eigen::VectorXd x, TFWState& curState)
{
	Eigen::VectorXd fullx = _projM.transpose() * x;

	int nverts = curState.basePos.rows();
	int nedges = _state.baseMesh.nEdges();

	for (auto& it : _clampedDOFs)
	{
		fullx(it.first) = it.second;
	}

	for (int i = 0; i < nverts; i++)
	{
		curState.amplitude(i) = fullx(i);
	}
	for (int i = 0; i < nedges; i++)
	{
		curState.dphi(i) = fullx(i + nverts);
	}
}


double TFWModel::value(const Eigen::VectorXd& x)
{
	convertVariables2CurState(x, _state);
	std::shared_ptr<TFWShell> reducedShell;
	double energy = 0;
	reducedShell = std::make_shared<TFWShell>(_setup, _state, _isUsePosHess, _isParallel);
	energy = reducedShell->elasticReducedEnergy(NULL, NULL);
	return energy;
}

double TFWModel::stretchingValue(const Eigen::VectorXd& x)
{
	convertVariables2CurState(x, _state);

	// std::cout<<"initial guess: "<<_state.amplitude(0)<<std::endl;
	std::shared_ptr<TFWShell> reducedShell;
	double energy = 0;

	reducedShell = std::make_shared<TFWShell>(_setup, _state, _isUsePosHess, _isParallel);
	reducedShell->_isPosHess = _isUsePosHess;
	energy = reducedShell->stretchingEnergy(NULL, NULL);
	return energy;
}

double TFWModel::bendingValue(const Eigen::VectorXd& x)
{
	convertVariables2CurState(x, _state);

	// std::cout<<"initial guess: "<<_state.amplitude(0)<<std::endl;
	std::shared_ptr<TFWShell> reducedShell;
	double energy = 0;

	reducedShell = std::make_shared<TFWShell>(_setup, _state, _isUsePosHess, _isParallel);
	reducedShell->_isPosHess = _isUsePosHess;
	energy = reducedShell->bendingEnergy(NULL, NULL);
	return energy;
}

void TFWModel::gradient(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
{
	std::shared_ptr<TFWShell> reducedShell;
	double energy = 0;
	convertVariables2CurState(x, _state);

	reducedShell = std::make_shared<TFWShell>(_setup, _state, _isUsePosHess, _isParallel);
	reducedShell->_isPosHess = _isUsePosHess;
	energy = reducedShell->elasticReducedEnergy(&grad, NULL);

	int nverts = _state.basePos.rows();
	grad = _projM * grad;
}

void TFWModel::hessian(const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& hessian)
{
	std::shared_ptr<TFWShell> reducedShell;
	double energy = 0;
	convertVariables2CurState(x, _state);
	Eigen::SparseMatrix<double> fullH;

	reducedShell = std::make_shared<TFWShell>(_setup, _state, _isUsePosHess, _isParallel);
	reducedShell->_isPosHess = _isUsePosHess;
	energy = reducedShell->elasticReducedEnergy(NULL, &fullH);


	hessian = _projM * fullH * _projM.transpose();
}

Eigen::VectorXd TFWModel::getProjectedGradient(const Eigen::VectorXd &x)
{
	Eigen::VectorXd g;
	Eigen::VectorXd Beq, BIneq;
	Eigen::SparseMatrix<double> Aeq, AIneq, I;
	Eigen::VectorXd lx, ux;

	gradient(x, g);
	return g;
}

void TFWModel::testValueAndGradient(const Eigen::VectorXd &x)
{
	std::cout << "Test value and gradient. " << std::endl;
	double f = value(x);
	Eigen::VectorXd grad;
	gradient(x, grad);
	Eigen::VectorXd dir = Eigen::VectorXd::Random(x.size());
	dir.normalize();
//    Eigen::VectorXd finiteDiff;
//    finiteGradient(x, finiteDiff);
//    std::cout<<grad - finiteDiff<<std::endl;
//    std::cout<<"Error norm: "<<(grad - finiteDiff).lpNorm<Eigen::Infinity>()<<std::endl;
//    std::cout<<std::endl<<"grad \t difference " <<std::endl;
//    for (int i = 0; i < grad.size(); i++)
//        std::cout << grad(i) << " " << grad(i) - finiteDiff(i) << std::endl;
	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		Eigen::VectorXd x1 = x + eps * dir;
		Eigen::VectorXd x2 = x - eps * dir;
		double f1 = value(x1);
		double f2 = value(x2);
		std::cout<<std::endl<<"eps: "<<eps<<std::endl;
		std::cout<<std::setprecision(std::numeric_limits<long double>::digits10 + 1)<<"energy: "<<f<<", energy after perturbation (right, left): "<<f1<<", "<<f2<<std::endl;
		std::cout<<std::setprecision(6);
		std::cout<<"right finite difference: "<<(f1 - f) / eps<<std::endl;
		std::cout<<"left finite difference: "<<(f - f2) / eps <<std::endl;
		std::cout<<"central difference: "<<(f1 - f2) / 2 / eps<<std::endl;
		std::cout<<"direction derivative: "<<grad.dot(dir)<<std::endl;
		std::cout<<"right error: "<<std::abs((f1 - f) / eps - grad.dot(dir))<<", left error: "<<std::abs((f - f2) / eps - grad.dot(dir))<<", central error: "<<std::abs((f1 - f2) / 2 / eps - grad.dot(dir))<<std::endl;
	}
}

void TFWModel::testGradientAndHessian(const Eigen::VectorXd& x)
{
	bool isUsePosHess = _isUsePosHess;
	_isUsePosHess = false;
	std::cout << "Test gradient and hessian. " << std::endl;
	Eigen::SparseMatrix<double> hess;
	Eigen::VectorXd deriv;
	gradient(x, deriv);
	hessian(x, hess);

	Eigen::VectorXd dir = Eigen::VectorXd::Random(x.size());
	dir.normalize();
	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		Eigen::VectorXd x1 = x + eps * dir;
		Eigen::VectorXd deriv1;
		gradient(x1, deriv1);

		std::cout << std::endl << "eps: " << eps << std::endl;
		std::cout << std::setprecision(6);
		std::cout << "finite difference: " << (deriv1 - deriv).norm() / eps << std::endl;
		std::cout << "direction derivative: " << (hess * dir).norm() << std::endl;
		std::cout << "error: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
	_isUsePosHess = isUsePosHess;
}


