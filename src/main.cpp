#include <iostream>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <TFWState.h>
#include <TFWSetup.h>
#include <TFWShell.h>
#include <TFWModel.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
TFW::TFWSetup setup;
TFW::TFWState state;
TFWModel model;


void loadSetup()
{
    // setup mesh 
	setup.PoissonsRatio = 1.0;
	setup.thickness = 0.01;
	setup.YoungsModulus = 1e5;
	setup.quadNum = 3;
    setup.restFlat = true;
    setup.clampedChosenVerts = true;
    setup.sffType = "midEdgeAve";
    setup.sff = std::make_shared<MidedgeAverageFormulation>();
    setup.restMeshPath = "../../../../openargus/examples/assets/square_mesh/square8x8.obj";
    igl::readOBJ(setup.restMeshPath, setup.restV, setup.restF);
    setup.buildRestFundamentalForms();
    setup.baseMeshPath = "../../../../openargus/scripts/debug/tfw_vs_ours/tfw_sheet/state/sheet_base.obj";
    Eigen::MatrixXi baseF;
    igl::readOBJ(setup.baseMeshPath, state.basePos, baseF);
    state.baseMesh = MeshConnectivity(baseF);
    setup.sff->initializeExtraDOFs(state.baseEdgeDOFs, state.baseMesh, state.basePos);
	locatePotentialPureTensionFaces(setup.abars, state.basePos, state.baseMesh.faces(), state.tensionFaces);
	state.computeBaseCurvature();           // This step is important. 

    setup.clampedDOFsPath = "";
    // read clamped dofs details
    // for each clamped vertex: 
    //setup.clampedDOFs[3 * vid + j] = state.basePos(vid, j);
    
	int nverts = state.basePos.rows();
	int nedges = state.baseMesh.nEdges();
    // load 
    state.amplitude = Eigen::VectorXd::Random(nverts);
    state.dphi = Eigen::VectorXd::Random(nedges);
    std::cout << "Nverts " << nverts << std::endl;
    std::cout << "Nedges " << nedges << std::endl;
	model.initialization(setup, state, NULL, "", true, false);
}
double getEnergy(Eigen::VectorXd* deriv)
{
	Eigen::VectorXd initX;
	model.convertCurState2Variables(state, initX);

	double energy = model.value(initX);
    // std::cout << "InitX " << initX << " Energy = " << energy << std::endl;
    if(deriv)
        model.gradient(initX, *deriv);
    return energy;
}
int main()
{
    std::cout << "Hello from main " << std::endl;
    loadSetup();   	
    Eigen::VectorXd deriv;
    double energy = getEnergy(&deriv);
    std::cout << "Energy = " << energy << std::endl;
    // std::cout << "Deriv " << deriv << std::endl;
    polyscope::init();
    polyscope::registerSurfaceMesh("Base Mesh", state.basePos, state.baseMesh.getF());
    polyscope::registerSurfaceMesh("Rest Mesh", setup.restV, setup.restF);
    polyscope::show();
}
