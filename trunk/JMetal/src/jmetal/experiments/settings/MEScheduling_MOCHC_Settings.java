package jmetal.experiments.settings;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Solution;
import jmetal.experiments.Settings;
import jmetal.metaheuristics.mochc.MOCHC_Int;
import jmetal.operators.crossover.Crossover;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.Mutation;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.scheduling.MEProblem;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.JMException;
import jmetal.util.listScheduling.RandomMCT;

public class MEScheduling_MOCHC_Settings extends Settings {
	public double initialConvergenceCount_;
	public double preservedPopulation_;
	public int convergenceValue_;
	public int populationSize_;
	public int maxEvaluations_;
	public double mutationProbability_;
	public double crossoverProbability_;

	/**
	 * Constructor
	 * 
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public MEScheduling_MOCHC_Settings(String problemName)
			throws ClassNotFoundException, IOException {
		super(problemName);

		String[] problemInfo = problemName.split(" ");
		assert (problemInfo.length == 4);
		String[] dimension = problemInfo[1].split("x");

		int taskCount = Integer.parseInt(dimension[0]);
		int machineCount = Integer.parseInt(dimension[1]);
		String scenarioPath = "/home/santiago/Scheduling/Energy-Makespan/instances.ruso/"
				+ problemInfo[1] + "/" + problemInfo[2];
		String workloadPath = "/home/santiago/Scheduling/Energy-Makespan/instances.ruso/"
				+ problemInfo[1] + "/" + problemInfo[3];

		problem_ = new MEProblem(taskCount, machineCount, scenarioPath,
				workloadPath);

		// Default settings
		initialConvergenceCount_ = 0.25;
		preservedPopulation_ = 0.05;
		convergenceValue_ = 3;
		populationSize_ = 100;
		maxEvaluations_ = 800000;
		mutationProbability_ = 0.35;
		crossoverProbability_ = 1.0;
	}

	/**
	 * Configure the MOCell algorithm with default parameter settings
	 * 
	 * @return an algorithm object
	 * @throws jmetal.util.JMException
	 */
	public Algorithm configure() throws JMException {
		Algorithm algorithm;

		Crossover crossover;
		Mutation mutation;
		Operator parentsSelection;
		Operator newGenerationSelection;

		QualityIndicator indicators;

		HashMap parameters; // Operator parameters
		algorithm = new MOCHC_Int(problem_);

		// Algorithm parameters
		algorithm.setInputParameter("initialConvergenceCount",
				initialConvergenceCount_);
		algorithm
				.setInputParameter("preservedPopulation", preservedPopulation_);
		algorithm.setInputParameter("convergenceValue", convergenceValue_);
		algorithm.setInputParameter("populationSize", populationSize_);
		algorithm.setInputParameter("maxEvaluations", maxEvaluations_);

		try {
			RandomMCT initMethod = new RandomMCT(problem_);
			Solution initSolution;

			List<Solution> population = new LinkedList<Solution>();
			for (int i = 0; i < populationSize_; i++) {
				initSolution = initMethod.compute();
				population.add(initSolution);
			}

			algorithm.setInputParameter("initialPopulation", population);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Mutation and Crossover for Real codification
		parameters = new HashMap();
		parameters.put("probability", crossoverProbability_);
		// parameters.put("distributionIndex", crossoverDistributionIndex_);
		/*
		 * crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover",
		 * parameters);
		 */
		crossover = CrossoverFactory.getCrossoverOperator(
				"SinglePointCrossover", parameters);

		parameters = new HashMap();
		parameters.put("probability", mutationProbability_);
		// parameters.put("distributionIndex", mutationDistributionIndex_);
		/*
		 * mutation = MutationFactory.getMutationOperator("PolynomialMutation",
		 * parameters);
		 */
		mutation = MutationFactory.getMutationOperator("BitFlipMutation",
				parameters);

		// Selection Operator
		parameters = null;

		parentsSelection = SelectionFactory.getSelectionOperator(
				"RandomSelection", parameters);
		/*
		 * parentsSelection = SelectionFactory.getSelectionOperator("BinaryTournament",
		 * parameters);
		 */
		parameters = new HashMap();
		parameters.put("problem", problem_);
		newGenerationSelection = SelectionFactory.getSelectionOperator(
				"RankingAndCrowdingSelection", parameters);

		// Add the operators to the algorithm
		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("cataclysmicMutation", mutation);
		algorithm.addOperator("parentSelection", parentsSelection);
		algorithm.addOperator("newGenerationSelection", newGenerationSelection);

		// Creating the indicator object
		if ((paretoFrontFile_ != null) && (!paretoFrontFile_.equals(""))) {
			indicators = new QualityIndicator(problem_, paretoFrontFile_);
			algorithm.setInputParameter("indicators", indicators);
		} // if

		return algorithm;
	} // configure
} // MOCell_Settings

