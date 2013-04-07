//  NSGAII_main.java
//
//  Author:
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

package jmetal.metaheuristics.nsgaII;

import jmetal.core.*;
import jmetal.operators.crossover.*;
import jmetal.operators.crossover.scheduling.MultiCoreHUXCrossover;
import jmetal.operators.mutation.*;
import jmetal.operators.mutation.scheduling.MultiCoreSwapMutation;
import jmetal.operators.selection.*;
import jmetal.problems.*;
import jmetal.problems.DTLZ.*;
import jmetal.problems.ZDT.*;
import jmetal.problems.scheduling.MultiCoreSchedulingProblem;
import jmetal.problems.WFG.*;
import jmetal.problems.LZ09.*;

import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.MersenneTwisterFast;
import jmetal.util.PseudoRandom;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.*;

import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.qualityIndicator.QualityIndicator;

/**
 * Class to configure and execute the NSGA-II algorithm.
 * 
 * Besides the classic NSGA-II, a steady-state version (ssNSGAII) is also
 * included (See: J.J. Durillo, A.J. Nebro, F. Luna and E. Alba "On the Effect
 * of the Steady-State Selection Scheme in Multi-Objective Genetic Algorithms"
 * 5th International Conference, EMO 2009, pp: 183-197. April 2009)
 */

public class NSGAII_OPI_main {
	public static Logger logger_; // Logger object
	public static FileHandler fileHandler_; // FileHandler object

	/**
	 * @param args
	 *            Command line arguments.
	 * @throws JMException
	 * @throws IOException
	 * @throws SecurityException
	 *             Usage: three options -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main problemName -
	 *             jmetal.metaheuristics.nsgaII.NSGAII_main problemName
	 *             paretoFrontFile
	 */
	public static void main(String[] args) throws JMException,
			SecurityException, IOException, ClassNotFoundException {

		Problem problem; // The problem to solve
		Algorithm algorithm; // The algorithm to use
		Operator crossover; // Crossover operator
		Operator mutation; // Mutation operator
		Operator selection; // Selection operator

		HashMap parameters; // Operator parameters

		QualityIndicator indicators; // Object to get quality indicators

		// Logger object and file to store log messages
		logger_ = Configuration.logger_;
		fileHandler_ = new FileHandler("NSGAII_opi_main.log");
		logger_.addHandler(fileHandler_);

		indicators = null;

		/*
		 * if (args.length == 1) { Object[] params = { "Real" }; problem = (new
		 * ProblemFactory()).getProblem(args[0], params); } // if else if
		 * (args.length == 2) { Object[] params = { "Real" }; problem = (new
		 * ProblemFactory()).getProblem(args[0], params); indicators = new
		 * QualityIndicator(problem, args[1]); } // if else { // Default problem
		 */
		if (args.length == 5) {
			String task_arrival_file = args[0];
			String task_priority_file = args[1];
			String task_cost_file = args[2];
			String task_cores_file = args[3];
			String machine_file = args[4];

			problem = loadMultiCoreScheduingProblem(task_arrival_file,
					task_priority_file, task_cores_file, task_cost_file,
					machine_file);

			algorithm = new NSGAII(problem);
			// algorithm = new ssNSGAII(problem);

			// Algorithm parameters
			algorithm.setInputParameter("populationSize", 100);
			algorithm.setInputParameter("maxEvaluations", 25000);

			// Mutation and Crossover for Real codification
			parameters = new HashMap();
			parameters.put("probability", 0.9);
			parameters.put("distributionIndex", 20.0);
			crossover = new MultiCoreHUXCrossover(parameters);

			parameters = new HashMap();
			parameters.put("probability", 1.0 / problem.getNumberOfVariables());
			parameters.put("distributionIndex", 20.0);
			mutation = new MultiCoreSwapMutation(parameters);

			// Selection Operator
			parameters = null;
			selection = SelectionFactory.getSelectionOperator(
					"BinaryTournament2", parameters);

			// Add the operators to the algorithm
			algorithm.addOperator("crossover", crossover);
			algorithm.addOperator("mutation", mutation);
			algorithm.addOperator("selection", selection);

			// Add the indicator object to the algorithm
			algorithm.setInputParameter("indicators", indicators);

			// Execute the Algorithm
			long initTime = System.currentTimeMillis();
			SolutionSet population = algorithm.execute();
			long estimatedTime = System.currentTimeMillis() - initTime;

			// Result messages
			logger_.info("Total execution time: " + estimatedTime + "ms");
			logger_.info("Variables values have been writen to file VAR");
			population.printVariablesToFile("VAR");
			logger_.info("Objectives values have been writen to file FUN");
			population.printObjectivesToFile("FUN");

			if (indicators != null) {
				logger_.info("Quality indicators");
				logger_.info("Hypervolume: "
						+ indicators.getHypervolume(population));
				logger_.info("GD         : " + indicators.getGD(population));
				logger_.info("IGD        : " + indicators.getIGD(population));
				logger_.info("Spread     : " + indicators.getSpread(population));
				logger_.info("Epsilon    : "
						+ indicators.getEpsilon(population));

				int evaluations = ((Integer) algorithm
						.getOutputParameter("evaluations")).intValue();
				logger_.info("Speed      : " + evaluations + " evaluations");
			} // if
		} else {
			logger_.severe("Usage error.");
			System.out
					.println("Input arguments: "
							+ " <task_arrival_file> <task_priority_file> <task_cost_file> <machine_cores_file> <machine_ssj_file> <machine_eidle_file> <machine_emax_file>");
		}
	} // main

	static MultiCoreSchedulingProblem loadMultiCoreScheduingProblem(
			String task_arrival_file, String task_priority_file,
			String task_cores_file, String task_cost_file, String machine_file) {

		logger_.info("task_arrival_file : " + task_arrival_file);
		logger_.info("task_priority_file: " + task_priority_file);
		logger_.info("task_cores_file   : " + task_cores_file);
		logger_.info("task_cost_file    : " + task_cost_file);
		logger_.info("machine_file      : " + machine_file);

		try {
			List<Integer> task_arrival = new ArrayList<Integer>();
			List<Integer> task_priority = new ArrayList<Integer>();
			List<Integer> task_cores = new ArrayList<Integer>();
			List<Double> task_cost = new ArrayList<Double>();

			List<String> input_files = new ArrayList<String>();
			input_files.add(task_priority_file);
			input_files.add(task_cores_file);
			
			List<List<Integer>> IntInputs = new ArrayList<List<Integer>>();
			IntInputs.add(task_priority);
			IntInputs.add(task_cores);

			for (int i = 0; i < IntInputs.size(); i++) {
				FileReader file = new FileReader(input_files.get(i));
				BufferedReader reader = new BufferedReader(file);

				String line;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						IntInputs.get(i).add(Integer.parseInt(line.trim()));
					}
				}

				reader.close();
				file.close();
			}
			
			{
				FileReader file = new FileReader(task_cost_file);
				BufferedReader reader = new BufferedReader(file);

				String line;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						task_cost.add(Double.parseDouble(line.trim()));
					}
				}

				reader.close();
				file.close();
			}
			
			{
				FileReader file = new FileReader(task_arrival_file);
				BufferedReader reader = new BufferedReader(file);

				String[] data;
				String line;
				Integer count;
				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						data = line.split("\t");
						count = Integer.parseInt(data[1].trim());
						
						for (int j = 0; j < count; j++) {
							task_arrival.add(Integer.parseInt(data[0].trim()));	
						}
					}
				}

				reader.close();
				file.close();
			}

			List<Integer> machine_cores = new ArrayList<Integer>();
			List<Integer> machine_ssj = new ArrayList<Integer>();
			List<Double> machine_eidle = new ArrayList<Double>();
			List<Double> machine_emax = new ArrayList<Double>();

			{
				FileReader file = new FileReader(machine_file);
				BufferedReader reader = new BufferedReader(file);

				String line;
				String[] data;

				while ((line = reader.readLine()) != null) {
					if (!line.isEmpty()) {
						data = line.split("\t");
						assert (data.length == 4);

						machine_cores.add(Integer.parseInt(data[0]));
						machine_ssj.add(Integer.parseInt(data[1]));
						machine_eidle.add(Double.parseDouble(data[2]));
						machine_emax.add(Double.parseDouble(data[3]));
					}
				}

				reader.close();
				file.close();
			}

			MultiCoreSchedulingProblem p = new MultiCoreSchedulingProblem(
					task_arrival, task_priority, task_cores, task_cost,
					machine_cores, machine_ssj, machine_eidle, machine_emax);
			
			return p;
		} catch (Exception e) {
			logger_.severe(e.toString());
			e.printStackTrace();
			System.exit(-1);
			return null;
		}
	}
} // NSGAII_main
