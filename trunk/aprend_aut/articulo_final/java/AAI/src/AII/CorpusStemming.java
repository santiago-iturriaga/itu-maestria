package AII;

import java.io.Reader;
import java.io.Writer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.OutputStream;

import org.tartarus.snowball.ext.spanishStemmer;

public class CorpusStemming {

	public static void main(String[] args) throws Throwable {
		if (args.length < 2) {
			System.out.println("Error!");
			System.exit(-1);
		}

		spanishStemmer stemmer = new spanishStemmer();

		Reader reader;
		reader = new InputStreamReader(new FileInputStream(args[1]));
		reader = new BufferedReader(reader);

		StringBuffer input = new StringBuffer();

		OutputStream outstream;
		outstream = System.out;

		Writer output = new OutputStreamWriter(outstream);
		output = new BufferedWriter(output);

		int repeat = 1;
		
		int character;
		while ((character = reader.read()) != -1) {
			char ch = (char) character;
			if (Character.isWhitespace((char) ch)) {
				if (input.length() > 0) {
					stemmer.setCurrent(input.toString());
					for (int i = repeat; i != 0; i--) {
						stemmer.stem();
					}
					output.write(stemmer.getCurrent());
					output.write('\n');
					input.delete(0, input.length());
				}
			} else {
				input.append(Character.toLowerCase(ch));
			}
		}
		output.flush();
	}

}
