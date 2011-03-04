package AII;

import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.io.Writer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.OutputStream;
import java.util.Scanner;

import org.tartarus.snowball.ext.spanishStemmer;

public class CorpusStemming {

	public static void main(String[] args) throws Throwable {
		if (args.length < 2) {
			System.out.println("Error!");
			System.exit(-1);
		}

		System.out.println("Corpus: " + args[0]);
		System.out.println("Stemmed corpus: " + args[1]);
		System.out.println("Stemming count: " + args[2]);
		
		spanishStemmer stemmer = new spanishStemmer();
		
		FileReader reader = new FileReader(args[0]);
		BufferedReader in = new BufferedReader(reader);

		FileWriter writer = new FileWriter(args[1]);
		BufferedWriter out = new BufferedWriter(writer);

		int repeat = Integer.parseInt(args[2]);
		
		String corpusLine;
        while ((corpusLine = in.readLine()) != null) {
        	corpusLine = corpusLine.trim();
        	
        	if (corpusLine.length() > 0) {
        		String[] tokens = corpusLine.split(" ");
        		String word = tokens[0];
        		
				stemmer.setCurrent(word);
				
				for (int i = repeat; i != 0; i--) {
					stemmer.stem();
				}
				
				out.write(stemmer.getCurrent());
				for (int i = 1; i < tokens.length; i++) {
					out.write(" " + tokens[i]);	
				}
				out.write("\n");
        	} else {
        		out.write("\n");
        	}        	
        }

        out.flush();
        out.close();
        in.close();
	}

}
