package AII;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

public class AdvCorpus {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String corpus = "";

		if (args.length == 1) {
			corpus = args[0];
		} else {
			System.exit(-1);
		}
		
		System.out.println("Corpus: " + corpus);

		try {
			FileReader reader = new FileReader(corpus);
			BufferedReader in = new BufferedReader(reader);

			ArrayList<String> currentLine = new ArrayList<String>();
			String corpusLine;
			boolean tieneCON_TILDE = false;
			
			while ((corpusLine = in.readLine()) != null) {
				String tokens;
				tokens = corpusLine.trim();

				String[] tokenArray;
				tokenArray = tokens.split(" ");
				
				if (tokenArray.length == 2) {
					currentLine.add(tokenArray[0]);
					if (tokenArray[1].equals("CON_TILDE")) {
						tieneCON_TILDE = true;
					}
				} else {
					if (tieneCON_TILDE) {
						for (int i = 0; i < currentLine.size(); i++) {
							System.out.print(currentLine.get(i) + " ");
						}
						System.out.print("\n\n\n");
					}
					tieneCON_TILDE = false;
					currentLine.clear();
				}
			}

			in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

}
