package justLR;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

import Utils.SUtils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class justLR {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances structure;

	int N;
	int n;
	int nc;
	int[] paramsPerAtt;

	int np = 0;
	int[] startPerAtt;

	boolean[] isNumericTrue;

	private double[] parameters;

	private boolean m_MVerb = false; 							 // -V
	private double m_Eta = 0.01;                                 // -E
	
	private double m_Lambda = 0.001;						// -L
	private boolean m_DoRegularization = false;			// -R
	private String m_O = "sgd";                                   	// -O
	
	private static final int BUFFER_SIZE = 100000;


	public void buildClassifier(File sourceFile) throws Exception {

		System.out.println("[----- justLR -----]: Reading structure -- " + sourceFile);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		System.out.println("Experiment Configuration");
		System.out.println(" ----------------------------------- ");
		System.out.println("m_O = " + m_O);
		System.out.println("m_DoRegularization = " + m_DoRegularization);
		if (m_DoRegularization) {
			System.out.println("m_Lambda = " + m_Lambda);
			
		}
		System.out.println("m_Eta = " + m_Eta);
		System.out.println(" ----------------------------------- ");

		// remove instances with missing class
		n = structure.numAttributes() - 1;
		nc = structure.numClasses();
		N = structure.numInstances();

		isNumericTrue = new boolean[n];
		paramsPerAtt = new int[n];

		for (int u = 0; u < n; u++) {
			if (structure.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = structure.attribute(u).numValues();
			} else if (structure.attribute(u).isNumeric()) {
				isNumericTrue[u] = true;
				paramsPerAtt[u] = 1;
			}
		}

		startPerAtt = new int[n];

		np = nc;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			if (structure.attribute(u).isNominal()) {
				np += (paramsPerAtt[u] * nc);
			} else if (structure.attribute(u).isNumeric()) {
				np += (1 * nc);
			}
		}

		parameters = new double[np];
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);

		/* 
		 * ------------------------------------------------------
		 * Optimization
		 * ------------------------------------------------------
		 */

		if (m_O.equalsIgnoreCase("sgd")) {

			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				double[] gradients = new double[np];

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);
				
				if (m_DoRegularization) {
					regularizeGradient(gradients);
				}

				double stepSize = m_Eta;
				for (int i = 0; i < np; i++) {
					parameters[i] = parameters[i] - stepSize * gradients[i];
				}

			}

		} else if (m_O.equalsIgnoreCase("adagrad")) {

			double smoothingParameter = 1e-9;

			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			double[] G = new double[np];

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				double[] gradients = new double[np];

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);
				
				if (m_DoRegularization) {
					regularizeGradient(gradients);
				}

				for (int j = 0; j < np; j++) {
					G[j] += ((gradients[j] * gradients[j]));
				}

				double stepSize[] = new double[np];
				for (int j = 0; j < np; j++) {
					stepSize[j] = m_Eta / (smoothingParameter + Math.sqrt(G[j]));

					if (stepSize[j] == Double.POSITIVE_INFINITY) {
						stepSize[j] = 0.0;
					}
				}

				for (int i = 0; i < np; i++) {
					parameters[i] = parameters[i] - stepSize[i] * gradients[i];
				}
			}


		} else if (m_O.equalsIgnoreCase("plr")) {

			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			double epsilon = 1e-9;

			double[] gbar = new double[np];
			double[] vbar = new double[np];
			double[] hbar = new double[np];
			double[] vpart = new double[np];
			double[] taus = new double[np];

			for (int i = 0; i < np; i++) {
				gbar[i] = 0;
				vbar[i] = 1.0 * epsilon;
				hbar[i] = 1.0;

				vpart[i] = (gbar[i] * gbar[i]) / vbar[i];
				taus[i] = (1.0 + epsilon) * 2;
			}

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				double[] gradients = new double[np];
				double[] hessians = new double[np];

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);
				computeHessian(row, probs, x_C, hessians);			
				
				if (m_DoRegularization) {
					regularizeGradient(gradients);
					regularizeHessian(hessians);
				}

				for (int j = 0; j < np; j++) {
					gbar[j] = (1 - 1/taus[j]) * gbar[j] + 1/taus[j] * gradients[j];
					vbar[j] = (1 - 1/taus[j]) * vbar[j] + 1/taus[j] * gradients[j] * gradients[j];
					hbar[j] = (1 - 1/taus[j]) * hbar[j] + 1/taus[j] * hessians[j];

					vpart[j] = 0;
					vpart[j] = vpart[j] + (gbar[j] * gbar[j]) / vbar[j];

					taus[j] = (1 - vpart[j]) * taus[j];
					taus[j] += (1 + epsilon);
				}

				double stepSize[] = new double[np];
				for (int j = 0; j < np; j++) {
					stepSize[j] = vpart[j] / (hbar[j] + epsilon);
				}

				for (int i = 0; i < np; i++) {
					parameters[i] = parameters[i] - stepSize[i] * gradients[i];
				}

			}
		}

	}

	public double[] distributionForInstance(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = parameters[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						probs[c] += (parameters[pos] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						probs[c] += parameters[pos];
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = parameters[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						probs[c] += (parameters[pos] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						probs[c] += parameters[pos];
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		for (int c = 0; c < nc; c++) {
			gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
		}

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int c = 0; c < nc; c++) {
					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * uval;
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
					}
				}
			}
		}

	}

	private void computeHessian(Instance inst, double[] probs, int x_C, double[] hessians) {

		double[] d =new double[nc];

		for (int c = 0;  c < nc; c++) {
			d[c] =  (1 - probs[c]) * probs[c];
		}

		for (int c = 0; c < nc; c++) {
			hessians[c] += d[c];
		}

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int c = 0; c < nc; c++) {
					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						hessians[pos] += d[c] * uval;
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						hessians[pos] += d[c];
					}
				}
			}
		}

	}

	public double regularizeFunction() {
		double f = 0.0;
		for (int i = 0; i < np; i++) {
			f += m_Lambda/2 * parameters[i] * parameters[i];
		}
		return f;
	}

	public void regularizeGradient(double[] grad) {
		for (int i = 0; i < np; i++) {
			grad[i] += m_Lambda * parameters[i];
		}
	}
	
	public void regularizeHessian(double[] hessians) {
		for (int i = 0; i < np; i++) {
			hessians[i] += m_Lambda;
		}
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);

		m_O = Utils.getOption('O', options);

		m_DoRegularization = Utils.getFlag('R', options); 

		String ML = Utils.getOption('L', options);
		if (ML.length() != 0) {
			m_Lambda = Double.parseDouble(ML);
		}

		Utils.checkForRemainingOptions(options);
	}

	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public int getNInstances() {
		return N;
	}

	public int getNc() {
		return nc;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}
	
	public void setM_MVerb(boolean m_MVerb) {
		this.m_MVerb = m_MVerb;
	}

	public String getM_O() {
		return m_O;
	}

	public void setM_O(String m_O) {
		this.m_O = m_O;
	}
	
	public boolean isM_DoRegularization() {
		return m_DoRegularization;
	}

	public void setM_DoRegularization(boolean m_DoRegularization) {
		this.m_DoRegularization = m_DoRegularization;
	}
	
	public double getM_Lambda() {
		return m_Lambda;
	}

	public void setM_Lambda(double m_Lambda) {
		this.m_Lambda = m_Lambda;
	}
	
	public double getM_Eta() {
		return m_Eta;
	}

	public void setM_Eta(double m_Eta) {
		this.m_Eta = m_Eta;
	}

	public int getnAttributes() {
		return n;
	}

	public int getNumericPosition(int u, int c) {
		return startPerAtt[u] + (paramsPerAtt[u] * c);
	}
	public int getNominalPosition(int u, int uval, int c) {
		return startPerAtt[u] + ((paramsPerAtt[u] * c) + uval);
	}

}
