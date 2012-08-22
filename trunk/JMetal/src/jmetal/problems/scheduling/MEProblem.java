package jmetal.problems.scheduling;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.IntSolutionType;
import jmetal.util.JMException;
import java.io.*;

public class MEProblem extends Problem {
	public class Scenario {
		public int[] Cores;
		public int[] SSJ;
		public float[] IdleEnergy;
		public float[] MaxEnergy;
	}

	public class Workload {
		public float[][] ETC;
		public float[][] Energy;
	}

	public int taskCount_;
	public int machineCount_;
	public Scenario scenario_;
	public Workload workload_;

	public MEProblem(int taskCount, int machineCount, String scenarioPath,
			String workloadPath) throws ClassNotFoundException, IOException {
		this.taskCount_ = taskCount;
		this.machineCount_ = machineCount;

		this.scenario_ = new Scenario();
		this.scenario_.Cores = new int[this.machineCount_];
		this.scenario_.SSJ = new int[this.machineCount_];
		this.scenario_.IdleEnergy = new float[this.machineCount_];
		this.scenario_.MaxEnergy = new float[this.machineCount_];

		FileInputStream fstream = new FileInputStream(scenarioPath);
		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		String strLine;

		int currentMachine = 0;
		int currentIndex;
		int currentData;

		int cores = 0;
		int ssj = 0;
		float idle = 0;
		float max = 0;
		String[] scenarioData;

		while ((strLine = br.readLine()) != null) {
			scenarioData = strLine.split(" |\t");

			currentData = 0;
			currentIndex = 0;
			while ((currentData < 4) && (currentIndex < scenarioData.length)) {
				if (!scenarioData[currentIndex].trim().equals("")) {
					if (currentData == 0) {
						cores = Integer.parseInt(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 1) {
						ssj = Integer.parseInt(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 2) {
						idle = Float.parseFloat(scenarioData[currentIndex]
								.trim());
					} else if (currentData == 3) {
						max = Float.parseFloat(scenarioData[currentIndex]
								.trim());
					}

					currentData++;
				}

				currentIndex++;
			}

			this.scenario_.Cores[currentMachine] = cores;
			this.scenario_.SSJ[currentMachine] = ssj;
			this.scenario_.IdleEnergy[currentMachine] = idle;
			this.scenario_.MaxEnergy[currentMachine] = max;
			currentMachine++;
		}
		in.close();

		this.workload_ = new Workload();
		this.workload_.ETC = new float[this.machineCount_][this.taskCount_];
		this.workload_.Energy = new float[this.machineCount_][this.taskCount_];

		fstream = new FileInputStream(workloadPath);
		in = new DataInputStream(fstream);
		br = new BufferedReader(new InputStreamReader(in));

		currentMachine = 0;
		int currentTask = 0;

		while ((strLine = br.readLine()) != null) {
			this.workload_.ETC[currentMachine][currentTask] = Float
					.parseFloat(strLine) / this.scenario_.SSJ[currentMachine];
			this.workload_.Energy[currentMachine][currentTask] = this.workload_.ETC[currentMachine][currentTask]
					* this.scenario_.MaxEnergy[currentMachine];

			currentMachine++;
			
			if (currentMachine == this.machineCount_) {
				currentTask++;
				currentMachine = 0;
			}
		}
		in.close();

		numberOfVariables_ = taskCount;
		numberOfObjectives_ = 2; /* makespan and energy */
		numberOfConstraints_ = 0;
		problemName_ = "MEProblem";

		upperLimit_ = new double[numberOfVariables_];
		lowerLimit_ = new double[numberOfVariables_];
		for (int var = 0; var < numberOfVariables_; var++) {
			lowerLimit_[var] = 0;
			upperLimit_[var] = machineCount-1;
		} // for

		solutionType_ = new IntSolutionType(this);
	}

	@Override
	public void evaluate(Solution solution) throws JMException {
		float totalEnergyConsumption = 0;

		float makespan = 0;
		float[] computeTime = new float[machineCount_];

		int assignedMachine;
		for (int t = 0; t < taskCount_; t++) {
			assignedMachine = (int) solution.getDecisionVariables()[t]
					.getValue();

			computeTime[assignedMachine] += workload_.ETC[assignedMachine][t];
			if (computeTime[assignedMachine] > makespan)
				makespan = computeTime[assignedMachine];

			totalEnergyConsumption += workload_.Energy[assignedMachine][t];
		}

		for (int m = 0; m < machineCount_; m++) {
			totalEnergyConsumption += (scenario_.IdleEnergy[m] * (makespan - computeTime[m]));
		}

		solution.setObjective(0, makespan);
		solution.setObjective(1, totalEnergyConsumption);
	}

	public void evaluateConstraints(Solution solution) throws JMException {
		/* no constraints defined */
	}
}
