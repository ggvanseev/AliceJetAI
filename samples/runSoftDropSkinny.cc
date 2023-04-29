#include <iostream>
#include <chrono>

#include "TFile.h"
#include "TTree.h"
#include "TMath.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "include/ProgressBar.h"

#include "PU14/EventMixer.hh"
#include "PU14/CmdLine.hh"
#include "PU14/PU14.hh"

#include "include/jetCollection.hh"
#include "include/softDropGroomer.hh"
#include "include/softDropCounter.hh"
#include "include/treeWriter.hh"

using namespace std;
using namespace fastjet;

// This class runs soft drop for signal jets (no background subtraction)

// ./runSoftDrop -hard PythiaEventsTune14PtHat120.pu14  -nev 1

int main (int argc, char ** argv) {

  auto start_time = std::chrono::steady_clock::now();
  
  CmdLine cmdline(argc,argv);
  // inputs read from command line
  int nEvent = cmdline.value<int>("-nev",1);  // first argument: command line option; second argument: default value
  //bool verbose = cmdline.present("-verbose");

  std::cout << "will run on " << nEvent << " events" << std::endl;

  // Uncomment to silence fastjet banner
  ClusterSequence::set_fastjet_banner_stream(NULL);

  //to write info to root tree
  treeWriter trwSig("jetTreeSig");
 
  //Jet definition
  double R                   = 0.4;
  double ghostRapMax         = 6.0;
  double ghost_area          = 0.005;
  int    active_area_repeats = 1;
  fastjet::GhostedAreaSpec ghost_spec(ghostRapMax, active_area_repeats, ghost_area);
  fastjet::AreaDefinition area_def = fastjet::AreaDefinition(fastjet::active_area,ghost_spec);
  fastjet::JetDefinition jet_def(antikt_algorithm, R);

  double jetRapMax = 3.0;
  fastjet::Selector jet_selector = SelectorAbsRapMax(jetRapMax);

  ProgressBar Bar(cout, nEvent);
  Bar.SetStyle(-1);

  EventMixer mixer(&cmdline);  //the mixing machinery from PU14 workshop

  // loop over events
  int iev = 0;
  unsigned int entryDiv = (nEvent > 200) ? nEvent / 200 : 1;
  while ( mixer.next_event() && iev < nEvent )
  {
    // increment event number    
    iev++;

    Bar.Update(iev);
    Bar.PrintWithMod(entryDiv);

    std::vector<fastjet::PseudoJet> particlesMerged = mixer.particles();

    std::vector<double> eventWeight;
    eventWeight.push_back(mixer.hard_weight());
    eventWeight.push_back(mixer.pu_weight());

    // extract hard partons that initiated the jets
    fastjet::Selector parton_selector = SelectorVertexNumber(-1);
    vector<PseudoJet> partons = parton_selector(particlesMerged);

    // select final state particles from hard event only
    fastjet::Selector sig_selector = SelectorVertexNumber(0);
    vector<PseudoJet> particlesSig = sig_selector(particlesMerged);
    
    //---------------------------------------------------------------------------
    //   jet clustering
    //---------------------------------------------------------------------------
    
    // run the clustering, extract the signal jets
    fastjet::ClusterSequenceArea csSig(particlesSig, jet_def, area_def);
    std::vector<fastjet::PseudoJet> jets = sorted_by_pt(jet_selector(csSig.inclusive_jets(10.)));
    jetCollection jetCollectionSig(jets);


    //---------------------------------------------------------------------------
    //   link initiator partons to jets
    //---------------------------------------------------------------------------

    double dR_cut = 0.2; 
    
    std::vector<double> dR_p1; // store dR values for each initiator parton
    std::vector<double> dR_p2;
    std::vector<double> jetInitPDG; // store PDG values of jet initiator

    // loop over jets to find dR values for each parton
    for(unsigned int i = 0; i < jets.size(); ++i) {
      dR_p1.push_back(partons[0].delta_R(jets[i]));
      dR_p2.push_back(partons[1].delta_R(jets[i]));
      jetInitPDG.push_back(TMath::QuietNaN());  // fill with NaN values initially
    }
    
    // find smallest dR values, store corresponding PDG values for each jet
    int min_p1 = std::distance(dR_p1.begin(), std::min_element(dR_p1.begin(), dR_p1.end()));
    int min_p2 = std::distance(dR_p2.begin(), std::min_element(dR_p2.begin(), dR_p2.end()));
    if(dR_p1[min_p1] <= dR_cut){
      jetInitPDG[min_p1] = partons[0].user_info<PU14>().pdg_id();
    }
    if(dR_p2[min_p2] <= dR_cut){
      jetInitPDG[min_p2] = partons[1].user_info<PU14>().pdg_id();
    }

    //jetCollectionSig.addVector("p1_jet_dR",     dR_p1);
    //jetCollectionSig.addVector("p2_jet_dR",     dR_p2);
    jetCollectionSig.addVector("jetInitPDG", jetInitPDG);

    //---------------------------------------------------------------------------
    //   Recursive Soft Drop for signal jets
    //---------------------------------------------------------------------------
    
    softDropCounter sdcSig(0.1, 0.0,R,0.0);
    sdcSig.setRecursiveAlgo(0);//0 = CA 1 = AKT 2 = KT
    sdcSig.run(jetCollectionSig);

    jetCollectionSig.addVector("sigJetRecur_jetpt",     sdcSig.getPts());
    jetCollectionSig.addVector("sigJetRecur_z",         sdcSig.getZgs());
    jetCollectionSig.addVector("sigJetRecur_dr12",      sdcSig.getDRs());
    //jetCollectionSig.addVector("sigJetRecur_erad",      sdcSig.getErads());
    //jetCollectionSig.addVector("sigJetRecur_logdr12",   sdcSig.getLog1DRs());
    //jetCollectionSig.addVector("sigJetRecur_logztheta", sdcSig.getLogzDRs());
    //jetCollectionSig.addVector("sigJetRecur_tf",        sdcSig.getTfs());
    jetCollectionSig.addVector("sigJetRecur_nSD",       sdcSig.calculateNSD(0.0));
    //jetCollectionSig.addVector("sigJetRecur_zSD",       sdcSig.calculateNSD(1.0));


    //---------------------------------------------------------------------------
    //   write tree
    //---------------------------------------------------------------------------
    
    //Give variable we want to write out to treeWriter.
    //Only vectors of the types 'jetCollection', and 'double', 'int', 'fastjet::PseudoJet' are supported

    trwSig.addCollection("eventWeight",   eventWeight);
    trwSig.addCollection("sigJet",        jetCollectionSig);
    
    // add hard partons to tree
    trwSig.addPartonCollection("partons",       partons);
    //trwSig.addCollection("parton_jet_dR",       parton_jet_dR);
    //trwSig.addCollection("parton_jet_link",     parton_jet_link);
    
    trwSig.fillTree();  //signal jets
  }//event loop

  Bar.Update(nEvent);
  Bar.Print();
  Bar.PrintLine();

  // EDITED: changed output file location
  TFile *fout = new TFile("JetToyHIResultSoftDropSkinny.root","RECREATE");
  trwSig.getTree()->Write();
  
  fout->Write();
  fout->Close();

  double time_in_seconds = std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::steady_clock::now() - start_time).count() / 1000.0;
  std::cout << "runFromFile: " << time_in_seconds << std::endl;
}
