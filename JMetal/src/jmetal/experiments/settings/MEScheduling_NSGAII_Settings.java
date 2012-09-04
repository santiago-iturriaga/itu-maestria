package jmetal.experiments.settings;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.experiments.Settings;
import jmetal.metaheuristics.nsgaII.NSGAII;
import jmetal.operators.crossover.Crossover;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.Mutation;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.Selection;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.scheduling.MEProblem;
import jmetal.problems.scheduling.MEProblem.Scenario;
import jmetal.problems.scheduling.MEProblem.Workload;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.JMException;
import jmetal.util.listScheduling.RandomMCT;

public class MEScheduling_NSGAII_Settings extends Settings {
	public int populationSize_;
	public int maxEvaluations_;
	public double mutationProbability_;
	public double crossoverProbability_;
	public double mutationDistributionIndex_;
	public double crossoverDistributionIndex_;

	public MEScheduling_NSGAII_Settings(String problemName)
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
		populationSize_ = 100;
		maxEvaluations_ = 500000;
		mutationProbability_ = 1.0 / problem_.getNumberOfVariables();
		crossoverProbability_ = 0.9;
		mutationDistributionIndex_ = 20.0;
		crossoverDistributionIndex_ = 20.0;
	}

	@Override
	public Algorithm configure() throws JMException {
		Algorithm algorithm;
		Selection selection;
		Crossover crossover;
		Mutation mutation;

		HashMap parameters; // Operator parameters

		QualityIndicator indicators;

		// Creating the algorithm. There are two choices: NSGAII and its steady-
		// state variant ssNSGAII
		algorithm = new NSGAII(problem_);
		// algorithm = new ssNSGAII(problem_) ;

		// Algorithm parameters
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
		parameters.put("distributionIndex", crossoverDistributionIndex_);
		// crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover",
		// parameters);
		crossover = CrossoverFactory.getCrossoverOperator(
				"SinglePointCrossover", parameters);

		parameters = new HashMap();
		parameters.put("probability", mutationProbability_);
		parameters.put("distributionIndex", mutationDistributionIndex_);
		// mutation = MutationFactory.getMutationOperator("PolynomialMutation",
		// parameters);
		mutation = MutationFactory.getMutationOperator("BitFlipMutation",
				parameters);

		// Selection Operator
		parameters = null;
		selection = SelectionFactory.getSelectionOperator("BinaryTournament2",
				parameters);

		// Add the operators to the algorithm
		algorithm.addOperator("crossover", crossover);
		algorithm.addOperator("mutation", mutation);
		algorithm.addOperator("selection", selection);

		// Creating the indicator object
		if ((paretoFrontFile_ != null) && (!paretoFrontFile_.equals(""))) {
			indicators = new QualityIndicator(problem_, paretoFrontFile_);
			algorithm.setInputParameter("indicators", indicators);
		} // if

		return algorithm;
	}
}
