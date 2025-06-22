#include <iostream>
#include <fstream>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <TFWState.h>
#include <TFWSetup.h>
#include <TFWShell.h>
#include <TFWModel.h>
#include <polyscope/polyscope.h>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <polyscope/surface_mesh.h>
TFW::TFWSetup setup;
TFW::TFWState state;
TFWModel model;

template<typename T>
bool openEigenData(std::string fileToOpen, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* data)
{
    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
    std::vector<T> matrixEntries;
    // try{
        // in this object we store the data from the matrix
        std::ifstream matrixDataFile(fileToOpen);
        // this variable is used to store the row of the matrix that contains commas 
        std::string matrixRowString;

        // this variable is used to store the matrix entry;
        std::string matrixEntry;

        // this variable is used to track the number of rows
        int matrixRowNumber = 0;


        while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
        {
            std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

            while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
            {
                matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
            }
            matrixRowNumber++; //update the column numbers
        }

        // here we convet the vector variable into the matrix and return the resulting object, 
        // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
        if(matrixRowNumber != 0)
            (*data) = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
        else
            return false;
        return true;
    // }
    // catch(const std::exception &e)
    // {
    // 	std::cerr << "Failed to read data! Return zero matrix\n";
    // 	data=  MatrixNN::Zero(0,0);
    // }
}

template<typename T>
void writeEigenData(std::string filename, T data)
{
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << data.format(CSVFormat);
}

void loadSetup()
{
    // setup mesh 
    setup.PoissonsRatio = 0.56;
    setup.thickness = 0.01;
    setup.YoungsModulus = 373.33;
    setup.quadNum = 3;
    setup.buildQuadraturePoints();
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
    // removes flat triangle amplitudes
    locatePotentialPureTensionFaces(setup.abars, state.basePos, state.baseMesh.faces(), state.tensionFaces);
    state.computeBaseCurvature();           // This step is important. 

    setup.clampedDOFsPath = "../../../../openargus/examples/assets/square_mesh/square8x82EdgesClamped.txt";
    Eigen::MatrixXi clampedverts;
    openEigenData(setup.clampedDOFsPath, &clampedverts);
    std::cout << "Clamped Verts " << std::endl;
    std::cout << clampedverts << std::endl;
    int ncv = clampedverts.rows();
    // read clamped dofs details
    for(int i=0; i<ncv; i++)
    {
        int vid = clampedverts(i);
        // for each clamped vertex: 
        //setup.clampedDOFs[3 * vid + j] = state.basePos(vid, j);
        for(int j=0; j<3; j++)
            setup.clampedDOFs[3*vid + j] = state.basePos(vid, j);
    }

    int nverts = state.basePos.rows();
    int nedges = state.baseMesh.nEdges();

    // load 
    Eigen::MatrixXd data;
    bool ld = openEigenData("../../../../openargus/scripts/debug/tfw_vs_ours/tfw_sheet/results/x0", &data);
    std::cout << data.rows() << " x " << data.cols() << std::endl;
    state.amplitude = data(Eigen::seq(0,80),0);
    state.dphi = data(Eigen::seq(81, 81 + 207),0);
    std::cout << "Nverts " << nverts << std::endl;
    std::cout << "Nedges " << nedges << std::endl;
    // fill clamped dofs info
    std::map<int, double> clampedDOFs;
	if (setup.clampedChosenVerts) // whether we clamped the vertices which were clamped in the TFT step
	{
		std::map<int, double>::const_iterator it;
		for (it = setup.clampedDOFs.begin(); it != setup.clampedDOFs.end(); it++)
		{
			state.amplitude[it->first / 3] = 0;
			clampedDOFs[it->first / 3] = 0;
		}
	}


    model.initialization(setup, state, &clampedDOFs, "", true, false);
}
// only saves wrinkle parametes
void saveState(std::string filepath)
{
    Eigen::VectorXd params;
    int nverts = state.basePos.rows();
    int nedges = state.baseMesh.nEdges();
    params.resize(nverts + nedges);
    params(Eigen::seq(0, nverts-1)) = state.amplitude;
    params(Eigen::seq(nverts, nverts + nedges -1)) = state.dphi;
    writeEigenData(filepath, params); 
}

double getEnergy(Eigen::VectorXd* deriv)
{
    Eigen::VectorXd initX;
    model.convertCurState2Variables(state, initX);

    double energy = model.value(initX);
    // std::cout << "InitX " << initX << " Energy = " << energy << std::endl;
    if(deriv)
        model.gradient(initX, *deriv);
    (*deriv) = model._projM.transpose() * (*deriv);
    return energy;
}
class Functor
{
    public:
    TFWModel m_model;
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {
        double energy = model.value(x);
        model.gradient(x, grad);
        return energy;
    }
};
void optimize()
{
    Eigen::VectorXd initX;
    model.convertCurState2Variables(state, initX);
    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    // Create solver and function object
    //<double> solver(param);
    LBFGSpp::LBFGSBSolver<double> solver(param);
    // Bounds
    int nverts = state.basePos.rows();
    int nedges = state.baseMesh.nEdges();
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(nverts + nedges, -std::numeric_limits<double>::max());
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(nverts + nedges, std::numeric_limits<double>::max());
    lb(Eigen::seq(0, nverts-1)) = Eigen::VectorXd::Constant(nverts, 0);
    lb = model._projM * lb;
    ub = model._projM * ub;
    double e;
    Functor f;
    f.m_model = model;
    // save initial state
    saveState("../../checkpoints/x0");
    // optimize
    for(int iter=1; iter<10000; iter++)
    {
        solver.minimize(f, initX, e, lb, ub);
        // load optimal values
        model.convertVariables2CurState(initX, state);
        // save state
        saveState("../../checkpoints/x" + std::to_string(iter));
    }
}

int main()
{
    loadSetup();   	

    Eigen::VectorXd deriv;
    double energy = getEnergy(&deriv);
    std::cout << "Energy = " << energy << std::endl;
    optimize();
    // std::cout << "Deriv \n" << deriv << std::endl;
    // std::fstream f("deriv.txt", std::ios_base::out);
    // f << deriv;
    // f.close();
    // polyscope::init();
    // polyscope::registerSurfaceMesh("Base Mesh", state.basePos, state.baseMesh.getF());
    // polyscope::registerSurfaceMesh("Rest Mesh", setup.restV, setup.restF);
    // polyscope::show();
    return 0;
}
