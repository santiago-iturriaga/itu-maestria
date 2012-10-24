//  PAES_Settings.java 
//
//  Authors:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package jmetal.experiments.settings;

import jmetal.metaheuristics.paes.*;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.Mutation;
import jmetal.operators.mutation.MutationFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.scheduling.MEProblem;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;

import jmetal.core.Algorithm;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.experiments.Settings;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.JMException;
import jmetal.util.Configuration.*;
import jmetal.util.listScheduling.RandomMCT;

/**
 * Settings class of algorithm PAES
 */
public class MEScheduling_PAES_Settings extends Settings {

	public int maxEvaluations_;
	public int archiveSize_;
	public int biSections_;
	public double mutationProbability_;
	public double distributionIndex_;

	/**
	 * Constructor
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 */
	public MEScheduling_PAES_Settings(String problemName) throws ClassNotFoundException, IOException {
		super(problemName);

		String[] problemInfo = problemName.split(" ");
		assert (problemInfo.length == 4);
		String[] dimension = problemInfo[1].split("x");

		int taskCount = Integer.parseInt(dimension[0]);
		int machineCount = Integer.parseInt(dimension[1]);
		String scenarioPath = "/home/santiago/Scheduling/Instances/Makespan-Energy/"
				+ problemInfo[1] + ".ME/" + problemInfo[2];
		String workloadPath = "/home/santiago/Scheduling/Instances/Makespan-Energy/"
				+ problemInfo[1] + ".ME/" + problemInfo[3];

		problem_ = new MEProblem(taskCount, machineCount, scenarioPath,
				workloadPath);
		
		// Default settings
		maxEvaluations_ = 850000*2;
		archiveSize_ = 100;
		biSections_ = 5;
		mutationProbability_ = 1.0 / problem_.getNumberOfVariables();
		distributionIndex_ = 20.0;
	} // PAES_Settings

	/**
	 * Configure the MOCell algorithm with default parameter settings
	 * 
	 * @return an algorithm object
	 * @throws jmetal.util.JMException
	 */
	public Algorithm configure() throws JMException {
		Algorithm algorithm;
		Mutation mutation;

		QualityIndicator indicators;

		HashMap parameters; // Operator parameters

		// Creating the problem
		algorithm = new PAES(problem_);

		// Algorithm parameters
		algorithm.setInputParameter("maxEvaluations", maxEvaluations_);
		algorithm.setInputParameter("biSections", biSections_);
		algorithm.setInputParameter("archiveSize", archiveSize_);

		try {
			RandomMCT initMethod = new RandomMCT(problem_);
			Solution initSolution;

			List<Solution> population = new LinkedList<Solution>();
			initSolution = initMethod.compute();
			population.add(initSolution);

			algorithm.setInputParameter("initialPopulation", population);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// Mutation (Real variables)
		parameters = new HashMap();
		parameters.put("probability", mutationProbability_);
		parameters.put("distributionIndex", distributionIndex_);
		/*mutation = MutationFactory.getMutationOperator("PolynomialMutation",
				parameters);*/
		mutation = MutationFactory.getMutationOperator("BitFlipMutation",
				parameters);

		// Mutation (BinaryReal variables)
		// mutation = MutationFactory.getMutationOperator("BitFlipMutation");
		// mutation.setParameter("probability",0.1);

		// Add the operators to the algorithm
		algorithm.addOperator("mutation", mutation);

		// Creating the indicator object
		if ((paretoFrontFile_ != null) && (!paretoFrontFile_.equals(""))) {
			indicators = new QualityIndicator(problem_, paretoFrontFile_);
			algorithm.setInputParameter("indicators", indicators);
		} // if
		return algorithm;
	} // configure
} // PAES_Settings
