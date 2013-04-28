//  NSGAII_Settings.java 
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

import java.util.HashMap;

import jmetal.metaheuristics.nsgaII.*;
import jmetal.operators.crossover.Crossover;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.crossover.scheduling.MultiCoreHUXCrossover;
import jmetal.operators.mutation.Mutation;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.mutation.scheduling.MultiCoreSwapMutation;
import jmetal.operators.selection.Selection;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;
import jmetal.core.*;
import jmetal.experiments.Settings;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.IRandomGenerator;
import jmetal.util.JMException;
import jmetal.util.MersenneTwisterFast;
import jmetal.util.PseudoRandom;

public class OPI_NSGAII_Settings extends Settings {
	public int populationSize_;
	public int maxEvaluations_;
	public double mutationProbability_;
	public double crossoverProbability_;
	public double mutationDistributionIndex_;
	public double crossoverDistributionIndex_;

	/**
	 * Constructor
	 * 
	 * @throws JMException
	 */
	public OPI_NSGAII_Settings(String problem) throws JMException {
		super(problem);

		String base_path = "";
		String task_arrival_file = "";
		String task_priority_file = "";
		String task_cost_file = "";
		String task_cores_file = "";
		String machine_file = "";

		if (problem.equals("FE-HCSP_8x2")) {
			// python convert_to_gams_m2.py 8 2 8x2/arrival.0
			// 8x2/priorities.0 8x2/workload_high.0 8x2/cores_c2.0
			// 8x2/scenario_c4_high.1 > gams_8x2_m2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c2.0";
			machine_file = "scenario_c4_high.1";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/";
		} else if (problem.equals("FE-HCSP_8x2_2")) {
			// python convert_to_gams_m2.py 8 2 8x2/arrival.1
			// 8x2/priorities.1 8x2/workload_high.1 8x2/cores_c4.15
			// 8x2/scenario_c6_high.6 > gams_8x2_m2.2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.15";
			machine_file = "scenario_c6_high.6";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/";
		} else if (problem.equals("FE-HCSP_8x2_3")) {
			// python convert_to_gams_m2.py 8 2 8x2/arrival.2
			// 8x2/priorities.2 8x2/workload_high.2 8x2/cores_c4.21
			// 8x2/scenario_c4_high.3 > gams_8x2_m2.3.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.21";
			machine_file = "scenario_c4_high.3";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/";
		} else if (problem.equals("FE-HCSP_16x3")) {
			// python convert_to_gams_m2.py 16 3 16x3/arrival.0
			// 16x3/priorities.0 16x3/workload_high.0 16x3/cores_c4.19
			// 16x3/scenario_c6_mid.31 > gams_16x3_m2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.19";
			machine_file = "scenario_c6_mid.31";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/";
		} else if (problem.equals("FE-HCSP_16x3_2")) {
			// python convert_to_gams_m2.py 16 3 16x3/arrival.1
			// 16x3/priorities.1 16x3/workload_high.1 16x3/cores_c4.10
			// 16x3/scenario_c6_high.1 > gams_16x3_m2.2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.10";
			machine_file = "scenario_c6_high.1";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/";
		} else if (problem.equals("FE-HCSP_16x3_3")) {
			// python convert_to_gams_m2.py 16 3 16x3/arrival.2
			// 16x3/priorities.2 16x3/workload_high.2 16x3/cores_c4.16
			// 16x3/scenario_c4_high.0 > gams_16x3_m2.3.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.16";
			machine_file = "scenario_c4_high.0";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/";
		} else if (problem.equals("FE-HCSP_32x4")) {
			// python convert_to_gams_m2.py 32 4 32x4/arrival.0
			// 32x4/priorities.0 32x4/workload_high.0 32x4/cores_c8.22
			// 32x4/scenario_c12_high.2 > gams_32x4_m2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c8.22";
			machine_file = "scenario_c12_high.2";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/32x4/";
		} else if (problem.equals("FE-HCSP_32x4_2")) {
			// python convert_to_gams_m2.py 32 4 32x4/arrival.1
			// 32x4/priorities.1 32x4/workload_high.1 32x4/cores_c8.21
			// 32x4/scenario_c10_mid.40 > gams_32x4_m2.2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c8.21";
			machine_file = "scenario_c10_mid.40";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/32x4/";
		} else if (problem.equals("FE-HCSP_32x4_3")) {
			// python convert_to_gams_m2.py 32 4 32x4/arrival.2
			// 32x4/priorities.2 32x4/workload_high.2 32x4/cores_c4.8
			// 32x4/scenario_c4_high.1 > gams_32x4_m2.3.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c4.8";
			machine_file = "scenario_c4_high.1";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/32x4/";
		} else if (problem.equals("FE-HCSP_512x16")) {
			// python convert_to_gams.py 512 16 512x16/arrival.0
			// 512x16/priorities.0 512x16/workload_high.0 512x16/cores_c8.1
			// 512x16/scenario_c12_mid.2 > gams_512x16.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c8.1";
			machine_file = "scenario_c12_mid.2";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/512x16/";
		} else if (problem.equals("FE-HCSP_512x16_2")) {
			// python convert_to_gams.py 512 16 512x16/arrival.1
			// 512x16/priorities.1 512x16/workload_high.1 512x16/cores_c8.27
			// 512x16/scenario_c12_low.8 > gams_512x16.2.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c8.27";
			machine_file = "scenario_c12_low.8";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/512x16/";
		} else if (problem.equals("FE-HCSP_512x16_3")) {
			// python convert_to_gams.py 512 16 512x16/arrival.2
			// 512x16/priorities.2 512x16/workload_high.2 512x16/cores_c8.10
			// 512x16/scenario_c10_mid.4 > gams_512x16.3.txt
			task_arrival_file = "arrival.0";
			task_priority_file = "priorities.0";
			task_cost_file = "workload_high.0";
			task_cores_file = "cores_c8.10";
			machine_file = "scenario_c10_mid.4";

			base_path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/512x16/";
		}

		problem_ = MultiCoreSchedulingProblem.loadMultiCoreSchedulingProblem(
				base_path + task_arrival_file, base_path + task_priority_file,
				base_path + task_cores_file, base_path + task_cost_file,
				base_path + machine_file);

		// Default settings
		populationSize_ = 100;
		maxEvaluations_ = 400000;
		mutationProbability_ = 1.0 / ((MultiCoreSchedulingProblem) problem_).NUM_TASKS;
		crossoverProbability_ = 0.9;
		mutationDistributionIndex_ = 20.0;
		crossoverDistributionIndex_ = 20.0;
	} // NSGAII_Settings

	/**
	 * Configure NSGAII with user-defined parameter settings
	 * 
	 * @return A NSGAII algorithm object
	 * @throws jmetal.util.JMException
	 */
	public Algorithm configure() throws JMException {
		Algorithm algorithm;
		Selection selection;
		Crossover crossover;
		Mutation mutation;

		HashMap parameters; // Operator parameters

		QualityIndicator indicators;

		// IRandomGenerator gen = new RandomGenerator();
		IRandomGenerator gen = new MersenneTwisterFast(1);
		PseudoRandom.setRandomGenerator(gen);
		
		// Creating the algorithm. There are two choices: NSGAII and its steady-
		// state variant ssNSGAII
		algorithm = new NSGAII(problem_);
		// algorithm = new ssNSGAII(problem_) ;

		// Algorithm parameters
		algorithm.setInputParameter("populationSize", populationSize_);
		algorithm.setInputParameter("maxEvaluations", maxEvaluations_);

		// Mutation and Crossover for Real codification
		parameters = new HashMap();
		parameters.put("probability", crossoverProbability_);
		parameters.put("distributionIndex", crossoverDistributionIndex_);
		//crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover", parameters);
		crossover = new MultiCoreHUXCrossover(parameters);

		parameters = new HashMap();
		parameters.put("probability", mutationProbability_);
		parameters.put("distributionIndex", mutationDistributionIndex_);
		//mutation = MutationFactory.getMutationOperator("PolynomialMutation", parameters);
		mutation = new MultiCoreSwapMutation(parameters);

		// Selection Operator
		parameters = null;
		selection = SelectionFactory.getSelectionOperator("BinaryTournament2",
				parameters);

		// Add the operators to the algorithm
		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("mutation", mutation);
		algorithm.addOperator("selection", selection);

		return algorithm;
	} // configure
} // NSGAII_Settings
