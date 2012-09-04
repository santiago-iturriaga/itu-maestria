package jmetal.experiments;

import java.io.IOException;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import jmetal.core.Algorithm;
import jmetal.experiments.settings.MEScheduling_MOCHC_Settings;
import jmetal.experiments.settings.MEScheduling_MOCell_Settings;
import jmetal.experiments.settings.MEScheduling_NSGAII_Settings;
import jmetal.experiments.settings.MEScheduling_PAES_Settings;
import jmetal.util.JMException;

public class MESchedulingStudy extends Experiment {

	@Override
	public void algorithmSettings(String problemName, int problemId,
			Algorithm[] algorithm) throws ClassNotFoundException {
		try {
			int numberOfAlgorithms = algorithmNameList_.length;
			HashMap[] parameters = new HashMap[numberOfAlgorithms];

			for (int i = 0; i < numberOfAlgorithms; i++) {
				parameters[i] = new HashMap();
			} // for

			if ((!paretoFrontFile_[problemId].equals(""))
					|| (paretoFrontFile_[problemId] == null)) {
				for (int i = 0; i < numberOfAlgorithms; i++) {
					parameters[i].put("paretoFrontFile_",
							paretoFrontFile_[problemId]);
				}
			} // if

			algorithm[0] = new MEScheduling_MOCHC_Settings(problemName).configure(parameters[0]);
			algorithm[1] = new MEScheduling_NSGAII_Settings(problemName).configure(parameters[1]);
			algorithm[2] = new MEScheduling_MOCell_Settings(problemName).configure(parameters[2]);
			algorithm[3] = new MEScheduling_PAES_Settings(problemName).configure(parameters[3]);
		} catch (IllegalArgumentException ex) {
			Logger.getLogger(MESchedulingStudy.class.getName()).log(
					Level.SEVERE, null, ex);
		} catch (IllegalAccessException ex) {
			Logger.getLogger(MESchedulingStudy.class.getName()).log(
					Level.SEVERE, null, ex);
		} catch (JMException ex) {
			Logger.getLogger(MESchedulingStudy.class.getName()).log(
					Level.SEVERE, null, ex);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
			Logger.getLogger(MESchedulingStudy.class.getName()).log(
					Level.SEVERE, null, e);
		}
	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws JMException
	 */
	public static void main(String[] args) throws JMException, IOException {
		MESchedulingStudy exp = new MESchedulingStudy();

		exp.experimentName_ = "MESchedulingStudy";
		exp.algorithmNameList_ = new String[] { "MOCHC", "NSGAII", "MOCell", "PAES" };
		/*exp.problemList_ = new String[] { "MEProblem 512x16 scenario.0 workload.0", "MEProblem 2048x64 scenario.0 workload.0" };*/
		exp.problemList_ = new String[] { "MEProblem 512x16 scenario.0 workload.0" };
		exp.paretoFrontFile_ = new String[] { "", "", "", "" };
		exp.indicatorList_ = new String[] {};
		exp.experimentBaseDirectory_ = "/home/santiago/workspace/JMetal/results/"
				+ exp.experimentName_;
		exp.paretoFrontDirectory_ = exp.experimentBaseDirectory_
				+ "/pareto_fronts";

		int numberOfAlgorithms = exp.algorithmNameList_.length;
		exp.algorithmSettings_ = new Settings[numberOfAlgorithms];

		exp.independentRuns_ = 1;

		// Run the experiments
		int numberOfThreads;
		exp.runExperiment(numberOfThreads = 1);

		// Generate latex tables (comment this sentence is not desired)
		// exp.generateLatexTables();

		// Configure the R scripts to be generated
		/*
		 * int rows; int columns; String prefix; String[] problems;
		 * 
		 * rows = 2; columns = 3; prefix = new String("Problems"); problems =
		 * new String[] { "ZDT1", "ZDT2", "ZDT3", "ZDT4", "DTLZ1", "WFG2" };
		 * 
		 * boolean notch; exp.generateRBoxplotScripts(rows, columns, problems,
		 * prefix, notch = true, exp); exp.generateRWilcoxonScripts(problems,
		 * prefix, exp);
		 */
	}

}
