package jmetal.util.listScheduling;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.problems.scheduling.MEProblem;

public abstract class ListScheduler {
	protected MEProblem p;

	public ListScheduler(Problem p) throws Exception {
		if (p.getClass().equals(MEProblem.class)) {
			this.p = (MEProblem) p;
		} else {
			throw new Exception("Problem is not a scheduling problem.");
		}
	}

	protected abstract Solution compute() throws Exception;
}
