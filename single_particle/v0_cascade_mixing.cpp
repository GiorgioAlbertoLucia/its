/*
    Routine for a mixing of pions and protons to obtain the background shape of the invariant mass distribution.
*/

#include <Riostream.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <cmath>

namespace treeHandling {

    void treeMerging(const char *inputFileName, const char *treeName, TFile *outputFile)
    {

        TFile *inputFile = TFile::Open(inputFileName, "READ");
        TList *treeList = new TList();
        TIter nextDir(inputFile->GetListOfKeys());
        TKey *key;
        while ((key = (TKey *)nextDir()))
        {
            std::cout << "Reading directory: " << key->GetName() << std::endl;
            TObject *obj = key->ReadObj();

            if (obj->InheritsFrom(TDirectory::Class()))
            {
                TDirectory *dir = (TDirectory *)obj;
                TTree *tmpTree = (TTree *)dir->Get(treeName);
                treeList->Add(tmpTree);
            }
            else
            {
                std::cerr << "Missing trees in directory: " << key->GetName() << std::endl;
            }
        }

        outputFile->cd();
        TTree *tree = TTree::MergeTrees(treeList);
        tree->Write();
        inputFile->Close();
    }

}   // namespace treeHandling

namespace physics {

    const float kMassPion = 0.13957; // GeV/c^2
    const float kMassProton = 0.93827; // GeV/c^2

    /**
     * Calculate the invariant mass of two particles given their momentum vectors and masses.
     * @param p1 Momentum vector of the first particle (p, eta, phi).
     * @param p2 Momentum vector of the second particle (p, eta, phi).
     * @param m1 Mass of the first particle.
     * @param m2 Mass of the second particle.
    */
    float invariantMass(const std::array<float, 3>& p1, const std::array<float, 3>& p2, float m1, float m2) {
        float px1 = p1[0] * cos(p1[2]);
        float py1 = p1[0] * sin(p1[2]);
        float pz1 = p1[0] * sinh(p1[1]);
        float e1 = sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + m1 * m1);
        float px2 = p2[0] * cos(p2[2]);
        float py2 = p2[0] * sin(p2[2]);
        float pz2 = p2[0] * sinh(p2[1]);
        float e2 = sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + m2 * m2);
        float eTot = e1 + e2;
        float pxTot = px1 + px2;
        float pyTot = py1 + py2;
        float pzTot = pz1 + pz2;
        return sqrt(eTot * eTot - (pxTot * pxTot + pyTot * pyTot + pzTot * pzTot));
    }

    float momentumMother(const std::array<float, 3>& p1, const std::array<float, 3>& p2) {
        float px1 = p1[0] * cos(p1[2]);
        float py1 = p1[0] * sin(p1[2]);
        float pz1 = p1[0] * sinh(p1[1]);
        float px2 = p2[0] * cos(p2[2]);
        float py2 = p2[0] * sin(p2[2]);
        float pz2 = p2[0] * sinh(p2[1]);
        return sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2) + (pz1 + pz2) * (pz1 + pz2));
    }

    float randomAngleRotation(const float phi) {
        float randomAngle = gRandom->Uniform(0, 2 * M_PI);
        if (phi + randomAngle > M_PI) {
            return phi + randomAngle - 2 * M_PI;
        } else if (phi + randomAngle < -M_PI) {
            return phi + randomAngle + 2 * M_PI;
        }
        return phi + randomAngle;
    }

}   // namespace physics

typedef struct Particle {
    float p;
    float eta;
    float phi;
    uint32_t itsClusterSize;
    uint8_t partId;

    void setBranchAddress(TTree* tree) {
        tree->SetBranchAddress("fP", &p);
        tree->SetBranchAddress("fEta", &eta);
        tree->SetBranchAddress("fPhi", &phi);
        tree->SetBranchAddress("fItsClusterSize", &itsClusterSize);
        tree->SetBranchAddress("fPartID", &partId);
    }
} Particle;

void fillParticlesFromTree(TTree* tree, std::vector<Particle>& pions, std::vector<Particle>& protons) {
    
    std::cout << "Filling particles from tree with " << tree->GetEntries() << " entries." << std::endl; 

    Particle particle;
    particle.setBranchAddress(tree);

    const int nEntries = tree->GetEntries();
    for (int i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (particle.partId == 2) { // pion
            pions.push_back(particle);
        } else if (particle.partId == 4) { // proton
            protons.push_back(particle);
        }
    }
}

void perforMixingRotation(const std::vector<Particle>& protons, const std::vector<Particle>& pions, 
                    TH2F* h2PInvariantMass, const int mixingDepth = 2) {

    std::cout << "Starting mixing with " << protons.size() << " protons and " << pions.size() << " pions." << std::endl;

    const int nPions = pions.size();
    for (int iproton = 0; iproton < protons.size(); ++iproton) {

        if (iproton % (protons.size()/100) == 0) {
            std::cout << "Processing proton " << iproton << " / " << protons.size() << " (" 
                      << iproton/protons.size()*100 << "%)" << std::endl;
        }
        
        const Particle& proton = protons[iproton];
        std::array<float, 3> p1 = {std::abs(proton.p), proton.eta, proton.phi};
        for (int i = 0; i < mixingDepth; ++i) {
            Particle pion = pions[gRandom->Integer(nPions)];
            while (pion.p * proton.p > 0) { // ensure unlike-sign
                pion = pions[gRandom->Integer(nPions)];
            }
            const float rotatedPhiPion = physics::randomAngleRotation(pion.phi);
            std::array<float, 3> p2 = {std::abs(pion.p), pion.eta, rotatedPhiPion};
            const float invMass = physics::invariantMass(p1, p2, physics::kMassProton, physics::kMassPion);
            const float p = physics::momentumMother(p1, p2);
            const float charge = proton.p > 0 ? 1 : -1;
            h2PInvariantMass->Fill(charge * p, invMass);
        }
    }
}

void perforMixingLikeSign(const std::vector<Particle>& protons, const std::vector<Particle>& pions, 
                    TH2F* h2PInvariantMass, const int mixingDepth = 2) {

    std::cout << "Starting mixing with " << protons.size() << " protons and " << pions.size() << " pions." << std::endl;

    const int nPions = pions.size();
    for (int iproton = 0; iproton < protons.size(); ++iproton) {

        if (iproton % (protons.size()/100) == 0) {
            std::cout << "Processing proton " << iproton << " / " << protons.size() << " (" 
                      << iproton/protons.size()*100 << "%)" << std::endl;
        }
        
        const Particle& proton = protons[iproton];
        std::array<float, 3> p1 = {std::abs(proton.p), proton.eta, proton.phi};
        for (int i = 0; i < mixingDepth; ++i) {
            Particle pion = pions[gRandom->Integer(nPions)];
            while (pion.p * proton.p < 0) { // ensure like-sign
                pion = pions[gRandom->Integer(nPions)];
            }
            std::array<float, 3> p2 = {std::abs(pion.p), pion.eta, pion.phi};
            const float invMass = physics::invariantMass(p1, p2, physics::kMassProton, physics::kMassPion);
            const float p = physics::momentumMother(p1, p2);
            const float charge = proton.p > 0 ? 1 : -1;
            h2PInvariantMass->Fill(charge * p, invMass);
        }
    }
}

void run_lambda_mixing(const bool mergeTrees = false) {

    const char * mergedInputFileName = "merged_trees/data_04_08_2025_merged.root";
    const char * treeName = "O2clsttable";
    
    if (mergeTrees) {
        const char * unmergedInputFileName = "/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root";
        TFile * mergedTreeFile = TFile::Open(mergedInputFileName, "RECREATE");
        treeHandling::treeMerging(unmergedInputFileName, treeName, mergedTreeFile);
        mergedTreeFile->Close();
    }

    TFile * inputFile = TFile::Open(mergedInputFileName, "READ");
    TTree * tree = (TTree *)inputFile->Get(treeName);
    std::vector<Particle> pions, protons;

    fillParticlesFromTree(tree, pions, protons);

    auto h2PInvariantMassRotation = new TH2F("h2PInvariantMassRotation", "Invariant Mass Distribution - Rotation strategy; #it{p}_{p#pi} (GeV/#it{c}); M_{p#pi} (GeV/c^{2});", 100, -5, 5, 50, 1.08, 1.18);
    perforMixingRotation(protons, pions, h2PInvariantMassRotation, 2);

    auto h2PInvariantMassLikeSign = new TH2F("h2PInvariantMassLikeSign", "Invariant Mass Distribution - Like-sign strategy; #it{p}_{p#pi} (GeV/#it{c}); M_{p#pi} (GeV/c^{2});", 100, -5, 5, 50, 1.08, 1.18);
    perforMixingLikeSign(protons, pions, h2PInvariantMassLikeSign, 2);

    const char * outputFileName = "output/v0_cascade_mixing.root";
    TFile * outputFile = TFile::Open(outputFileName, "RECREATE");
    outputFile->cd();
    h2PInvariantMassRotation->Write();
    h2PInvariantMassLikeSign->Write();
    outputFile->Close();

}
