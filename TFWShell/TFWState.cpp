#include "TFWState.h"

#include "CommonFunctions.h"

using namespace TFW;

void TFWState::reinitializeWrinkleVaribles(const TFWSetup& setup)
{
	std::set<int> clampedVerts;
	clampedVerts.clear();
	std::map<int, double>::const_iterator it;
	for (it = setup.clampedDOFs.begin(); it != setup.clampedDOFs.end(); it++)
	{
		int vid = it->first / 3;
		if (clampedVerts.find(vid) == clampedVerts.end() && setup.clampedChosenVerts)
			clampedVerts.insert(vid);
	}

	std::cout << "Reinitialize amp and dphi." << std::endl;
	// estimateWrinkleVariablesFromStrainCutbyTension(setup.abars, basePos, baseMesh.faces(), clampedVerts, 0.01 * (basePos.maxCoeff() - basePos.minCoeff()), amplitude, phi, dphi, tensionFaces);
    // setup initialization
    //
	dualAmp.resize(0);
	dualDphi.resize(0);
}
