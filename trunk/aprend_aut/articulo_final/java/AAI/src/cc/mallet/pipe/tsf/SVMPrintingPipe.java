/* Copyright (C) 2003 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */
package cc.mallet.pipe.tsf;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.*;
import cc.mallet.util.Maths;
import cc.mallet.util.PropertyList;

/**
 * Created: Jul 6, 2005
 * 
 * @author <A HREF="mailto:casutton@cs.umass.edu>casutton@cs.umass.edu</A>
 * @version $Id: SequencePrintingPipe.java,v 1.1 2007/10/22 21:37:58 mccallum
 *          Exp $
 */
public class SVMPrintingPipe extends Pipe implements Serializable {
	private String logFile;
	private PrintWriter writer;
	private int index;

	public SVMPrintingPipe() {
		super(new Alphabet(), null);

		this.index = 0;
	}

	public SVMPrintingPipe(Alphabet dataDict) {
		super(dataDict, null);

		this.index = 0;
	}

	public SVMPrintingPipe(String logFile) throws FileNotFoundException {
		super(new Alphabet(), null);

		this.index = 0;
		this.logFile = logFile;
		this.writer = new PrintWriter(logFile);
	}

	public Instance pipe(Instance carrier) {
		Sequence data = (Sequence) carrier.getData();
		Sequence target = (Sequence) carrier.getTarget();

		if (data.size() != target.size())
			throw new IllegalArgumentException(
					"Trying to print into SimpleTagger format, where data and target lengths do not match\n"
							+ "data.length = "
							+ data.size()
							+ ", target.length = " + target.size());

		int N = data.size();

		if (data instanceof TokenSequence) {
			throw new UnsupportedOperationException("Not yet implemented.");
		} else if (data instanceof FeatureVectorSequence) {
			FeatureVectorSequence fvs = (FeatureVectorSequence) data;
			Alphabet dict = (fvs.size() > 0) ? fvs.getFeatureVector(0)
					.getAlphabet() : null;

			for (int i = 0; i < N; i++) {
				FeatureVector fv = fvs.getFeatureVector(i);

				int wordPos = -1;
				for (int currentPos = 0; (currentPos < fv.numLocations())
						&& (wordPos == -1); currentPos++) {
					String currentFeature = dict
							.lookupObject(
									fvs.getFeatureVector(i).indexAtLocation(
											currentPos)).toString();

					if (currentFeature.startsWith("WORD=")) {
						wordPos = currentPos;
					}
				}

				if (wordPos == -1) {
					throw new IllegalArgumentException();
				} else {
					writer.print(dict
							.lookupObject(
									fvs.getFeatureVector(i).indexAtLocation(
											wordPos)).toString()
							.substring("WORD=".length()) + ' ');
				}

				Object label = target.get(i);
				writer.print(label);

				for (int loc = 0; loc < fv.numLocations(); loc++) {
					if (loc != wordPos) {
						writer.print(' ');
						String fname = dict.lookupObject(
								fv.indexAtLocation(loc)).toString();
						double value = fv.valueAtLocation(loc);
						if (!Maths.almostEquals(value, 1.0)) {
							throw new IllegalArgumentException(
									"Printing to SimpleTagger format: FeatureVector not binary at time slice "
											+ i + " fv:" + fv);
						}
						writer.print(fname);
					}
				}
				writer.println();
			}
		} else {
			throw new IllegalArgumentException(
					"Don't know how to print data of type " + data);
		}

		// writer.println();
		writer.flush();
		return carrier;
	}

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeObject(logFile);

		writer.flush();
		writer.close();
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		logFile = (String) in.readObject();

		this.writer = new PrintWriter(logFile);
	}
}
