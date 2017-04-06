package justLR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.BitSet;

import org.apache.commons.math3.random.MersenneTwister;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class TwoFoldXValOOCFupla {

	private static String data = "";

	private static boolean m_MVerb = false; 							 // -V

	private static String m_O = "sgd";                                   	// -O

	private static boolean m_DoRegularization = false;			// -R
	private static double m_Lambda = 0.001;						// -L
	private static double m_Eta = 0.01;                                 // -E
	
	private static int m_NumIterations = 1;                           // -I
	private static int m_BufferSize = 1;                           // -B

	private static Instances instances = null;
	private static int m_nExp = 5;

	public static final int BUFFER_SIZE = 10*1024*1024; 	//100MB

	public static void main(String[] args) throws Exception {

		System.out.println("Executing TwoFoldXValOOCFupla");

		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File sourceFile;
		sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

		/*
		 * Read file sequentially, 10000 instances at a time
		 */
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);

		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int N = getNumData(sourceFile, structure);
		System.out.println("Read " + N + " datapoints");

		double m_RMSE = 0;
		double m_Error = 0;
		int NTest = 0;
		long seed = 3071980;

		/*
		 * Start m_nExp rounds of Experiments
		 */

		int lineNo = 0;
		Instance current;
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		while ((current = reader.readInstance(structure)) != null) {
			lineNo++;
		}

		double[][] instanceProbs = new double[lineNo][nc];

		for (int exp = 0; exp < m_nExp; exp++) {

			if (m_MVerb) {
				System.out.println("Experiment No. " + exp);
			}

			MersenneTwister rg = new MersenneTwister(seed);
			BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);

			// ---------------------------------------------------------
			// Train on Fold 0
			// ---------------------------------------------------------

			justLR learner = new justLR();
			learner.setM_MVerb(m_MVerb);
			learner.setM_O(m_O);
			learner.setM_Eta(m_Eta);
			learner.setM_DoRegularization(m_DoRegularization);
			if (m_DoRegularization) {
				learner.setM_Lambda(m_Lambda);
			}
			learner.setM_NumIterations(m_NumIterations);
			learner.setM_BufferSize(m_BufferSize);

			// creating tempFile for train0
			File trainFile = createTrainTmpFile(sourceFile, structure, test0Indexes);
			System.out.println("Train file generated");

			if (m_MVerb) {
				System.out.println("Training fold 0: trainFile is '" + trainFile.getAbsolutePath() + "'");
			}

			learner.buildClassifier(trainFile);

			// ---------------------------------------------------------
			// Test on Fold 1
			// ---------------------------------------------------------
			if (m_MVerb) {
				System.out.println("Testing fold 0 started");
			}

			int thisNTest = 0;

			lineNo = 0;
			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
			while ((current = reader.readInstance(structure)) != null) {
				if (test0Indexes.get(lineNo)) {
					double[] probs = new double[nc];
					probs = learner.distributionForInstance(current);
					int x_C = (int) current.classValue();

					// ------------------------------------
					// Update Error and RMSE
					// ------------------------------------
					int pred = -1;
					double bestProb = Double.MIN_VALUE;
					for (int y = 0; y < nc; y++) {
						if (!Double.isNaN(probs[y])) {
							if (probs[y] > bestProb) {
								pred = y;
								bestProb = probs[y];
							}
							m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
						} else {
							System.err.println("probs[ " + y + "] is NaN! oh no!");
						}
					}

					if (pred != x_C) {
						m_Error += 1;
					}

					thisNTest++;
					NTest++;

					instanceProbs[lineNo][pred]++;
				}
				lineNo++;
			}

			if (m_MVerb) {
				System.out.println("Testing fold 0 finished - 0-1=" + (m_Error / NTest) + "\trmse=" + Math.sqrt(m_RMSE / NTest));
			}

			if (Math.abs(thisNTest - test0Indexes.cardinality()) > 1) {
				System.err.println("no! " + thisNTest + "\t" + test0Indexes.cardinality());
			}

			BitSet test1Indexes = new BitSet(lineNo);
			test1Indexes.set(0, lineNo);
			test1Indexes.xor(test0Indexes);

			// ---------------------------------------------------------
			// Train on Fold 1
			// ---------------------------------------------------------
			learner = new justLR();
			learner.setM_MVerb(m_MVerb);
			learner.setM_O(m_O);
			learner.setM_Eta(m_Eta);
			learner.setM_DoRegularization(m_DoRegularization);
			if (m_DoRegularization) {
				learner.setM_Lambda(m_Lambda);
			}
			learner.setM_NumIterations(m_NumIterations);
			learner.setM_BufferSize(m_BufferSize);

			// creating tempFile for train0
			trainFile = createTrainTmpFile(sourceFile, structure, test1Indexes);

			if (m_MVerb) {
				System.out.println("Training fold 1: trainFile is '" + trainFile.getAbsolutePath() + "'");
			}

			learner.buildClassifier(trainFile);

			// ---------------------------------------------------------
			// Test on Fold 0
			// ---------------------------------------------------------
			if (m_MVerb) {
				System.out.println("Testing fold 0 started");
			}

			lineNo = 0;
			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
			while ((current = reader.readInstance(structure)) != null) {
				if (test1Indexes.get(lineNo)) {
					double[] probs = new double[nc];
					probs = learner.distributionForInstance(current);
					int x_C = (int) current.classValue();

					// ------------------------------------
					// Update Error and RMSE
					// ------------------------------------
					int pred = -1;
					double bestProb = Double.MIN_VALUE;
					for (int y = 0; y < nc; y++) {
						if (!Double.isNaN(probs[y])) {
							if (probs[y] > bestProb) {
								pred = y;
								bestProb = probs[y];
							}
							m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
						} else {
							System.err.println("probs[ " + y + "] is NaN! oh no!");
						}
					}

					if (pred != x_C) {
						m_Error += 1;
					}

					NTest++;

					instanceProbs[lineNo][pred]++;
				}
				lineNo++;
			}

			if (m_MVerb) {
				System.out.println("Testing exp " + exp + " fold 1 finished - 0-1=" + (m_Error / NTest) + "\trmse=" + Math.sqrt(m_RMSE / NTest));
			}

			seed++;
		} // Ends No. of Experiments


		double m_Bias = 0;
		double m_Sigma = 0;
		double m_Variance = 0;

		lineNo = 0;
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		while ((current = reader.readInstance(structure)) != null) {
			double[] predProbs = instanceProbs[lineNo];

			double pActual, pPred;
			double bsum = 0, vsum = 0, ssum = 0;
			for (int j = 0; j < nc; j++) {
				pActual = (current.classValue() == j) ? 1 : 0;
				pPred = predProbs[j] / m_nExp;
				bsum += (pActual - pPred) * (pActual - pPred) - pPred * (1 - pPred) / (m_nExp - 1);
				vsum += (pPred * pPred);
				ssum += pActual * pActual;
			}
			m_Bias += bsum;
			m_Variance += (1 - vsum);
			m_Sigma += (1 - ssum);

			lineNo++;
		}

		m_Bias = m_Bias / (2 * lineNo);
		m_Variance = (m_Error / NTest) - m_Bias;

		System.out.print("\nBias-Variance Decomposition\n");
		System.out.print( "\nData File   : " + data);
		System.out.print("\nError           : " + Utils.doubleToString(m_Error / NTest, 6, 4));
		System.out.print("\nBias^2        : " + Utils.doubleToString(m_Bias, 6, 4));
		System.out.print("\nVariance     : " + Utils.doubleToString(m_Variance, 6, 4));
		System.out.print("\nRMSE          : " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
		System.out.print("\n\n\n");

	}

	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static void setOptions(String[] options) throws Exception {

		String Strain = Utils.getOption('t', options);
		if (Strain.length() != 0) {
			data = Strain;
		}

		m_MVerb = Utils.getFlag('V', options);

		String strX = Utils.getOption('X', options);
		if (strX.length() != 0) {
			m_nExp = Integer.valueOf(strX);
		}
		
		String strI = Utils.getOption('I', options);
		if (strI.length() != 0) {
			m_NumIterations = Integer.valueOf(strI);
		}
		
		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_BufferSize = Integer.valueOf(strB);
		}
		
		String Soutput = Utils.getOption('O', options);
		if (Soutput.length() != 0) {
			m_O = Soutput;
		}

		m_DoRegularization = Utils.getFlag('R', options);

		if (m_DoRegularization) {
			String strL = Utils.getOption('L', options);
			if (strL.length() != 0) {
				m_Lambda = Double.valueOf(strL);
			}
		}
		
		String strE = Utils.getOption('E', options);
		if (strE.length() != 0) {
			m_Eta = Double.valueOf(strE);
		}
		

		Utils.checkForRemainingOptions(options);

	}

	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if(nLines%1000000==0){
				System.out.println(nLines);
			}
			nLines++;
		}
		return nLines;
	}

	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg) throws FileNotFoundException, IOException {
		BitSet res = new BitSet();
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if (rg.nextBoolean()) {
				res.set(nLines);
			}
			nLines++;
		}

		int expectedNLines = (nLines % 2 == 0) ? nLines / 2 : nLines / 2 + 1;
		int actualNLines = res.cardinality();

		if (actualNLines < expectedNLines) {
			while (actualNLines < expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (res.get(chosen));
				res.set(chosen);
				actualNLines++;
			}
		} else if (actualNLines > expectedNLines) {
			while (actualNLines > expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (!res.get(chosen));
				res.clear(chosen);
				actualNLines--;
			}
		}
		return res;
	}

	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
		File out = File.createTempFile("train-", ".arff");
		out.deleteOnExit();
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);

		Instance current;
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (!testIndexes.get(lineNo)) {
				fileSaver.writeIncremental(current);
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		return out;
	}

	public static String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

}
