/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/** 
 Count the number of times the provided regular expression matches
 the token text, and add a feature with the provided name having
 value equal to the count.

 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package cc.mallet.pipe.tsf;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import cc.mallet.pipe.*;
import cc.mallet.types.*;

public class InQuestionMarks extends Pipe implements Serializable {
	Pattern regex;
	Pattern destPatternRegex;
	String feature;

	public InQuestionMarks(String featureName, Pattern regex,
			Pattern destPatternRegex) {
		this.feature = featureName;
		this.regex = regex;
		this.destPatternRegex = destPatternRegex;
	}

	public Instance pipe(Instance carrier) {
		TokenSequence ts = (TokenSequence) carrier.getData();
		int count = 0;

		for (int i = 0; i < ts.size(); i++) {
			count = 0;
			Token t = ts.get(i);

			Matcher matcher = regex.matcher(t.getText());
			while (matcher.find()) {
				count++;
				break;
			}
		}

		if (count > 0) {
			for (int i = 0; i < ts.size(); i++) {
				count = 0;
				Token t = ts.get(i);

				Matcher matcher = destPatternRegex.matcher(t.getText());
				while (matcher.find()) {
					t.setFeatureValue(feature, 1);
				}
			}
		}

		return carrier;
	}

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeInt(CURRENT_SERIAL_VERSION);
		out.writeObject(regex);
		out.writeObject(destPatternRegex);
		out.writeObject(feature);
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		int version = in.readInt();
		regex = (Pattern) in.readObject();
		destPatternRegex = (Pattern) in.readObject();
		feature = (String) in.readObject();
	}
}
