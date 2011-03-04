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
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

public class SplitCorpus {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String corpus = "";
		String path = "";

		if (args.length == 2) {
			corpus = args[0];
			
			path = args[1];
			if (path.charAt(path.length()-1) == '/') {
				path = path.substring(0, path.length()-1);
			}
		} else {
			System.exit(-1);
		}

		System.out.println("Corpus: " + corpus);
	
		try {		
			FileReader reader = new FileReader(corpus);
			BufferedReader in = new BufferedReader(reader);
			
			FileWriter trainWriter;
			BufferedWriter trainBuffWriter;
			
			FileWriter testWriter;
			BufferedWriter testBuffWriter;

			FileWriter testWriterFull;
			BufferedWriter testBuffWriterFull;
			
			String corpusLine;
			int totalCorpusLines = 0;
			
            while ((corpusLine = in.readLine()) != null) {
				if (corpusLine.trim().equals("")) {
					totalCorpusLines++;
				}
            }

            in.close();

    		double fraction = 0.1;
            int linesFraction = (int) Math.round(fraction * totalCorpusLines); 
           
			System.out.println(">> Total lines: " + totalCorpusLines);
			System.out.println(">> Test fraction: " + linesFraction);
			
			for (int i = 0; i < 10; i++) {
				reader = new FileReader(corpus);
				in = new BufferedReader(reader);

				trainWriter = new FileWriter(path + "/train_" + i + ".txt",false);
				trainBuffWriter = new BufferedWriter(trainWriter);

				testWriter = new FileWriter(path + "/test_" + i + ".txt",false);
				testBuffWriter = new BufferedWriter(testWriter);

				testWriterFull = new FileWriter(path + "/test_full_" + i + ".txt",false);
				testBuffWriterFull = new BufferedWriter(testWriterFull);
				
				int start_offset = i * linesFraction;
				int end_offset;
				if (i < 9) {
					end_offset = start_offset + linesFraction;
				} else {
					end_offset = totalCorpusLines;
				}
				
				System.out.println("=====================================");
				System.out.println(">> Index: " + i);
				System.out.println(">> Start offset: " + start_offset);
				System.out.println(">> End offset: " + end_offset);
				
				ArrayList<String> currentLine = new ArrayList<String>();
				int corpusLineIndex = 0;
	            while ((corpusLine = in.readLine()) != null) {
	            	currentLine.add(corpusLine);
	            	
					if (corpusLine.trim().equals("")) {
						if ((corpusLineIndex >= start_offset)&&(corpusLineIndex <= end_offset)) {
							System.out.println(">> " + currentLine.size() + " a test");
							
							for (int j = 0; j < currentLine.size(); j++) {
								String tokens;
								tokens = currentLine.get(j).trim();
								
								String[] tokenArray;
								tokenArray = tokens.split(" ");
								
								if (tokenArray.length > 1) {
									testBuffWriter.write(tokenArray[0] + "\n");
									testBuffWriterFull.write(tokens + "\n");
								} else {
									testBuffWriter.write(tokens + "\n");
									testBuffWriterFull.write(tokens + "\n");	
								}
							}
						} else {
							System.out.println(">> " + currentLine.size() + " a train");
							
							for (int j = 0; j < currentLine.size(); j++) {
								trainBuffWriter.write(currentLine.get(j) + "\n");
							}
						}
						
						corpusLineIndex++;
						currentLine = new ArrayList<String>();
					}
	            }
	            
	            testBuffWriter.close();
	            testBuffWriterFull.close();
	            trainBuffWriter.close();
	            in.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

}
