package jmetal.util.listScheduling;

import jmetal.core.Problem;
import jmetal.core.Solution;

public class MCT extends RandomMCT {

	public MCT(Problem p) throws Exception {
		super(p);
		// TODO Auto-generated constructor stub
	}

	public Solution compute() throws Exception {
		return compute(0, 1);
	}

}
