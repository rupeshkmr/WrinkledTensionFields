#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "CommonFunctions.h"

namespace TFW
{
	class TFWSetup
	{
	public:
		TFWSetup()
		{
			restV.resize(0, 0);
			restF.resize(0, 0);
			clampedDOFs.clear();
			restEdgeDOFs.resize(0);

			thickness = 0;
			YoungsModulus = 0;
			PoissonsRatio = 0;

			quadPoints.clear();
			abars.clear();
			bbars.clear(); 
			
			clampedChosenVerts = true;
			restFlat = true;
			quadNum = 0;
			sffType = "";
			restMeshPath = "";
			baseMeshPath = "";
			clampedDOFsPath = "";
			phiPath = "";
			dphiPath = "";
			ampPath = "";

		}

	public:
		// Core data structures
		Eigen::MatrixXd restV;
		Eigen::MatrixXi restF; // mesh vertices of the original (unstitched) state

		std::map<int, double> clampedDOFs;
		double thickness;
		double YoungsModulus;
		double PoissonsRatio;

		int quadNum;
		std::string sffType;
		
		bool restFlat;
		bool clampedChosenVerts;
		std::shared_ptr<SecondFundamentalFormDiscretization> sff;
		Eigen::VectorXd restEdgeDOFs;

		std::vector<QuadraturePoints> quadPoints;
		
		
		// Derived from the above
		std::vector<Eigen::Matrix2d> abars;
		std::vector<Eigen::Matrix2d> bbars;

	public:	// file pathes
		std::string restMeshPath, baseMeshPath, clampedDOFsPath;
		std::string phiPath, dphiPath, ampPath;

		void buildQuadraturePoints();
		void buildRestFundamentalForms();
	};
}
