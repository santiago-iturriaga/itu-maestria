package jmetal.experiments.settings;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.experiments.Settings;
import jmetal.metaheuristics.mocell.MOCell;
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

public class MEScheduling_MOCell_Settings extends Settings {
	public int populationSize_;
	public int maxEvaluations_;
	public int archiveSize_;
	public int feedback_;
	public double mutationProbability_;
	public double crossoverProbability_;
	public double crossoverDistributionIndex_;
	public double mutationDistributionIndex_;

	/**
	 * Constructor
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 */
	public MEScheduling_MOCell_Settings(String problemName) throws ClassNotFoundException, IOException {
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
		populationSize_ = 100;
		maxEvaluations_ = 550000*2;
		archiveSize_ = 100;
		feedback_ = 20;
		mutationProbability_ = 1.0 / problem_.getNumberOfVariables();
		crossoverProbability_ = 0.9;
		crossoverDistributionIndex_ = 20.0;
		mutationDistributionIndex_ = 20.0;
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
		Operator selection;

		QualityIndicator indicators;

		HashMap parameters; // Operator parameters

		// Selecting the algorithm: there are six MOCell variants
		// algorithm = new sMOCell1(problem_) ;
		// algorithm = new sMOCell2(problem_) ;
		// algorithm = new aMOCell1(problem_) ;
		// algorithm = new aMOCell2(problem_) ;
		// algorithm = new aMOCell3(problem_) ;
		algorithm = new MOCell(problem_);

		// Algorithm parameters
		algorithm.setInputParameter("populationSize", populationSize_);
		algorithm.setInputParameter("maxEvaluations", maxEvaluations_);
		algorithm.setInputParameter("archiveSize", archiveSize_);
		algorithm.setInputParameter("feedBack", feedback_);

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
		/*crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover",
				parameters);*/
		crossover = CrossoverFactory.getCrossoverOperator(
				"SinglePointCrossover", parameters);

		parameters = new HashMap();
		parameters.put("probability", mutationProbability_);
		parameters.put("distributionIndex", mutationDistributionIndex_);
		/*mutation = MutationFactory.getMutationOperator("PolynomialMutation",
				parameters);*/
		mutation = MutationFactory.getMutationOperator("BitFlipMutation",
				parameters);

		// Selection Operator
		parameters = null;
		selection = SelectionFactory.getSelectionOperator("BinaryTournament",
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
	} // configure
} // MOCell_Settings

