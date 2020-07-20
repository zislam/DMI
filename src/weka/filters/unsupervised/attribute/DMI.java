/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    DMI.java
 *    Copyright (C) 2018 Michael Furner
 *
 */
package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.EMImputation;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * <!-- globalinfo-start -->
 * Class that implements the DMI imputation algorithm for imputing missing
 * values in a dataset. DMI splits the dataset into horizontal segments using a
 * C4.5 (J48) decision tree in order to increase the correlation between
 * attributes for EMI. EMI is performed to impute missing numerical attribute
 * values and mean/mode (within a leaf) imputation is used to perform missing
 * categorical attribute values. Uses Amri Napolitano's EMI implementation for
 * Weka.
 * <p/>
 * DMI specification from:
 * <p/>
 * Rahman, M. G., and Islam, M. Z. (2013): Missing Value Imputation Using
 * Decision Trees and Decision Forests by Splitting and Merging Records: Two
 * Novel Techniques, Knowledge-Based Systems, Vol. 53, pp. 51 - 65, ISSN
 * 0950-7051, DOI information: 10.1016/j.knosys.2013.08.023, Available at
 * http://www.sciencedirect.com/science/article/pii/S0950705113002591
 * <p/>
 * Changes:
 * <ul>
 * <li>Leaves that are too small to run EMI are replaced by the nearest node
 * above them in a tree. We call this process "merging".</li>
 * <li>Records that cannot be assigned to any leaves in a tree for imputing an
 * attribute will be assigned to the leaf with the closest centroid.</li>
 * <li>If there are too few records in the whole dataset to ever run EMI (i.e.
 * number of records < (number of numeric attributes in dataset + 2)) no merging takes place.</li>
 * </ul> <p/> <!-- globalinfo-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 * <p/>
 *
 * <pre> -D
 * minCategoriesForDiscretization - Minimum number of categories for discretization</pre>
 *
 * <pre> -N
 * j48MinRecordsInLeaf - Minimum number of records in a J48 tree leaf. A negative value will default to (number of numeric attributes in dataset + 2).</pre>
 *
 * <pre> -F
 * j48ConfidenceFactor - Confidence factor for J48</pre>
 *
 * <pre> -E
 * minRecordsForEMI - Minimum records in a leaf for EMI to be able to run. A negative value will default to (number of numeric attributes in dataset + 2)</pre>
 *
 * <pre> -I
 * emiNumIterations - Iterations for EMI. A negative value will be set to Integer.MAX_VALUE</pre>
 *
 * <pre> -L
 * emiLogLikelihoodThreshold - Log likelihood threshold for terminating EMI.</pre>
 * <!-- options-end -->
 *
 * @author Michael Furner
 * @version 0.2
 */
public class DMI extends DecisionTreeImputation {

    static final long serialVersionUID = -124098210938L;

    /**
     * Set to true if there aren't enough records in the dataset for EMI to be
     * usable
     */
    private boolean m_noMergeNoEMI = false;

    /**
     * Set to true if there are absolutely no complete records in the dataset
     */
    private boolean m_noComplete = false;
    
    /**
     * Minimum records in leaf for J48 Values below zero will be set to
     * (number of numeric attributes in dataset + 2) Defaults to -1.
     */
    private int m_j48MinRecordsInLeaf = -1;

    /**
     * Confidence factor for J48
     */
    private float m_j48ConfidenceFactor = 0.25f;

    /**
     * Minimum Categories for Discretization
     */
    private int m_minCategoriesForDiscretization = 2;

    /**
     * User value for minimum number of records in a leaf for EMI to run. If
     * this is not met, then missing values in that leaf will be imputed by
     * mean/mode imputation. Values below zero will be set to
     * (number of numeric attributes in dataset + 2) Defaults to -1.
     */
    private int m_minRecordsForEMI = -1;

    /**
     * Threshold for convergence of EM
     */
    private double m_emiLogLikelihoodThreshold = 1e-4;

    /**
     * Number of EM iterations. Any negative value is set to maximum integer
     * value.
     */
    private int m_emiNumIterations = -1;

    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        //runFilter(new DMI(), argv);
        try {
            DataSource ds = new DataSource("/home/michael/sync/RA/Weka SiMI/Zahid Adult/adult.arff");
            Instances data = ds.getDataSet();
            
            DMI dmi = new DMI();
            dmi.setInputFormat(data);
            Instances test = dmi.process(data);
            
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Determines the output format based on the input format and returns this.
     *
     * @param inputFormat the input format to base the output format on
     * @return the output format
     * @see #hasImmediateOutputFormat()
     * @see #batchFinished()
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return inputFormat;
    }

    /**
     * Perform DMI on given dataset
     *
     * @param input - dataset to process
     * @return Imputed dataset
     * @throws Exception
     */
    @Override
    protected Instances process(Instances input) {

        int tmpj48Rec = m_j48MinRecordsInLeaf;
        int tmpMinEMI = m_minRecordsForEMI;
        
        int numNumericAttributes = 0;
        for (int i = 0; i < input.numAttributes(); i++) {
            if (input.attribute(i).isNumeric()) {
                numNumericAttributes++;
            }
        }

        /* Set up minimum records */
        if (m_j48MinRecordsInLeaf < 0) {
            m_j48MinRecordsInLeaf = numNumericAttributes + 2;
        }
        if (m_minRecordsForEMI < 0) {
            m_minRecordsForEMI = numNumericAttributes + 2;
        }

        /* Step 1: Divide a full data set DF into two sub data sets Dc and Di */
        m_dataset = input;
        try {
            separateMissingRecords();
        } catch (Exception ex) {
            Logger.getLogger(DMI.class.getName()).log(Level.SEVERE, null, ex);
        }

        
        //if there aren't any complete records at all we may as well just run EMI on the dataset and return it
        if(m_datasetComplete.isEmpty()) {
            m_noComplete = true;
            m_noMergeNoEMI = true;
        }
        else if (m_datasetComplete.size() < numNumericAttributes) { //if this is the case we can't ever use EMI, and there's also no point merging
            m_noMergeNoEMI = true;
        }

        /* Step 2: Build a set of decision trees on Dc.*/
        // determine which attributes have missing values 
        ArrayList<Integer> attributesWithMissing = new ArrayList<Integer>();
        for (int i = 0; i < m_dataset.numAttributes(); i++) {
            if (m_dataset.attributeStats(i).missingCount > 0) {
                attributesWithMissing.add(i);
            }
        }

        // Iterate over the missing attributes, creating a copy of the dataset
        // and building a decision tree with each missing attribute as the  
        // class attribute 
        ArrayList<J48> treeList = new ArrayList<J48>();
        ArrayList<ArrayList<Instances>> datasetList = new ArrayList<ArrayList<Instances>>();
        ArrayList<ArrayList<String>> ruleList = new ArrayList<ArrayList<String>>();
        
        for (Integer i : attributesWithMissing) {
            
            ArrayList<String> newRules;
            if(!m_noComplete) {    
                Instances datasetCompleteCopy = new Instances(m_datasetComplete);
                int wekaAttrIndex = i + 1;

                // if this attribute is numerical, discretize it
                if (m_dataset.attribute(i).isNumeric()) {
                    Discretize discretize = new Discretize();

                    discretize.setAttributeIndices("" + wekaAttrIndex);

                    double range = m_dataset.attributeStats(i).numericStats.max
                            - m_dataset.attributeStats(i).numericStats.min;

                    if ((int) Math.sqrt(range) >= m_minCategoriesForDiscretization) {
                        discretize.setBins((int) Math.sqrt(range));
                    } else {
                        discretize.setBins(m_minCategoriesForDiscretization);
                    }

                    try{
                        discretize.setInputFormat(datasetCompleteCopy);
                        datasetCompleteCopy = Filter.useFilter(datasetCompleteCopy, discretize);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }

                datasetCompleteCopy.setClassIndex(i);
                if (datasetCompleteCopy.attributeStats(i).distinctCount < 2) {
                    newRules = new ArrayList<>();
                    newRules.add("all");
                } else {

                    //build the tree
                    J48 Ti = new J48();
                    Ti.setMinNumObj(m_j48MinRecordsInLeaf);
                    Ti.setConfidenceFactor(m_j48ConfidenceFactor);
                    Ti.setNumDecimalPlaces(10);
                    try {
                        Ti.buildClassifier(datasetCompleteCopy);
                    } catch (Exception ex) {
                        Logger.getLogger(DMI.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    double Si = Ti.measureNumLeaves();

                    //add these trees, datasets, and rules to their corresponding lists
                    treeList.add(Ti);
                    newRules = ruleExtractor(Ti.toString(), (int) Si);

                    //merge down the rules that are too small
                    if (!m_noMergeNoEMI) {
                        newRules = mergeSmallRules(newRules);
                    }
                }
            }
            else {
                newRules = new ArrayList<>();
                newRules.add("all");
            }

            ruleList.add(newRules);
            
            if(!m_noComplete) {
                datasetList.add(datasetsFromRules(m_datasetComplete, newRules));

            }
            else {
                datasetList.add(datasetsFromRules(m_dataset, newRules));
            }

        } //end step 2 for loop

        //Calculate mean record for each of the datasets
        ArrayList<ArrayList<Instance>> meanRecords = new ArrayList<ArrayList<Instance>>();
        for (int i = 0; i < datasetList.size(); i++) {

            ArrayList<Instance> meanSetForTree = new ArrayList<>();

            for (int j = 0; j < datasetList.get(i).size(); j++) {
                Instances thisDs = datasetList.get(i).get(j);
                DenseInstance meanInstance = new DenseInstance(thisDs.get(0));

                for (int k = 0; k < thisDs.numAttributes(); k++) {
                    if (thisDs.attribute(k).isNumeric()) {

                        double normalised = (thisDs.meanOrMode(k) - m_dataset.attributeStats(k).numericStats.min)
                                / (m_dataset.attributeStats(k).numericStats.max - m_dataset.attributeStats(k).numericStats.min);
                        meanInstance.setValue(k, normalised);

                    } else {
                        meanInstance.setValue(k, thisDs.meanOrMode(k));
                    }

                }

                meanSetForTree.add(meanInstance);

            } //end dataset loop

            meanRecords.add(meanSetForTree);

        } //end datasetlist loop

        //end calculate mean for each of the datasets
        /* Step 3: Assign each record of DI to the leaf where it falls in. 
                   Impute categorical missing values. */
        Instances m_datasetMissingCopy = new Instances(m_datasetMissing);
        for (int i = 0; i < m_datasetMissing.numInstances(); i++) {
            Instance r_i = m_datasetMissing.get(i);
            Instance r_iChanged = m_datasetMissingCopy.get(i);

            int attrCount = 0;
            for (Integer j : attributesWithMissing) {
  
                if (r_i.isMissing(j)) {

                    ArrayList<String> correspondingTree = ruleList.get(attrCount);
                    int correspondingRule = ruleGivenRecord(correspondingTree, r_i);

                    if (correspondingRule == -1) {
                        correspondingRule = closestMean(r_i, meanRecords.get(j));
                    }

                    if (m_datasetMissing.attribute(j).isNumeric()) {

                        //add this record to the corresponding rule sub ds
                        datasetList.get(attrCount).get(correspondingRule).add(r_i);

                    } else {

                        //Impute A_j of r_i by the majority value of A_j in d_j and update D_I
                        r_iChanged.setValue(j, datasetList.get(attrCount).get(correspondingRule).meanOrMode(j));

                    }

                } //end if j is missing
                attrCount++;

            } //end attr loop

        } //end DI record loop

        /* Step 4: Impute numerical missing values using EMI algorithm within 
                   the leaves. */
        /* set up list of imputed leaves, so that we only ever impute the sub ds
           for a leaf once. This corresponds to the variable I in the paper */
        ArrayList<ArrayList<Boolean>> imputedList = new ArrayList<ArrayList<Boolean>>();
        for (int i = 0; i < datasetList.size(); i++) {
            imputedList.add(new ArrayList<Boolean>());
            for (int j = 0; j < datasetList.get(i).size(); j++) {
                imputedList.get(i).add(false);
            }
        }

        /* Identify the indexes for numerical attributes after the categorical 
           ones are removed */
        HashMap<Integer, Integer> numericalOnlyIndexes = new HashMap<Integer, Integer>();
        for (int j = 0; j < m_dataset.numAttributes(); j++) {
            if (m_dataset.attribute(j).isNumeric()) {
                int newindex = 0;
                for (int i = 0; i < j; i++) {
                    if (m_dataset.attribute(i).isNumeric()) {
                        newindex++;
                    }
                }
                numericalOnlyIndexes.put(j, newindex);
            }
        }

        //perform imputations of numeric values
        for (int i = 0; i < m_datasetMissing.numInstances(); i++) {
            Instance r_i = m_datasetMissing.get(i);
            Instance r_iChanged = m_datasetMissingCopy.get(i);
            boolean anm = allNumericMissing(r_i, numericalOnlyIndexes.keySet().toArray());

            int attrCount = 0;
            for (Integer j : attributesWithMissing) {

                ArrayList<String> correspondingTree = ruleList.get(attrCount);

                if (r_i.isMissing(j) && m_datasetMissing.attribute(j).isNumeric()) {

                    int correspondingRule = ruleGivenRecord(correspondingTree, r_i);

                    if (!imputedList.get(attrCount).get(correspondingRule)) { //if we haven't imputed yet

                        Instances subDs = datasetList.get(attrCount).get(correspondingRule);
                        subDs = new Instances(subDs);
                        subDs.setClassIndex(j);
                        subDs.deleteAttributeType(Attribute.NOMINAL);
                        subDs.deleteAttributeType(Attribute.STRING);
                        subDs.deleteAttributeType(Attribute.RELATIONAL);
                        subDs.deleteAttributeType(Attribute.DATE);

                        //if we don't meet the requirements for EMI we have to use
                        //a mean / mode imputation.
                        //if(subDs.numInstances() >= subDs.numAttributes() + 2) {
                        if (!m_noMergeNoEMI && subDs.numAttributes() > 2) {
                            EMImputation emi = new EMImputation();
                            emi.setNumIterations(m_emiNumIterations);
                            emi.setLogLikelihoodThreshold(m_emiLogLikelihoodThreshold);
                            try{
                                emi.setInputFormat(subDs);
                                subDs = Filter.useFilter(subDs, emi);
                                
                                ReplaceMissingValues rmv = new ReplaceMissingValues();
                                String[] tmp = {"-unset-class-temporarily"};
                                rmv.setOptions(tmp);
                                rmv.setInputFormat(subDs);
                                
                                subDs = Filter.useFilter(subDs, rmv);
                            } catch(Exception e) {
                                e.printStackTrace();
                            }
                        } // }
                        else {
                            try{
                                ReplaceMissingValues rmv = new ReplaceMissingValues();
                                String[] tmp = {"-unset-class-temporarily"};
                                rmv.setOptions(tmp);
                                rmv.setInputFormat(subDs);
                                
                                subDs = Filter.useFilter(subDs, rmv);
                                
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }

                        datasetList.get(attrCount).get(correspondingRule).clear();
                        datasetList.get(attrCount).get(correspondingRule).addAll(subDs);

                        imputedList.get(attrCount).set(correspondingRule, true);
                    }
                    
                    //If all the numerical attributes in the record are missing we just use means
                    if(anm) {
                        
                            r_iChanged.setValue(j, 
                                    (meanRecords.get(attrCount).get(correspondingRule).value(j)     
                                        * (m_dataset.attributeStats(j).numericStats.max - m_dataset.attributeStats(j).numericStats.min)) + m_dataset.attributeStats(j).numericStats.min
                            );
                        
                        attrCount++;
                        continue;
                    }
                    
                    //we have already imputed this, get the result
                    //first find the right record in the imputed dataset
                    Instances dsToCheck = datasetList.get(attrCount).get(correspondingRule);
                    double theVal = Double.NaN;
                    for (int k = 0; k < dsToCheck.size(); k++) {

                        boolean correctRec = true;
                        
                        for (int kj = 0; kj < m_dataset.numAttributes(); kj++) {
                            if (kj != j) {

                                if (m_dataset.attribute(kj).isNumeric()) {
                                    if(!r_i.isMissing(kj)) {
                                        if (r_i.value(kj) != dsToCheck.get(k).value(numericalOnlyIndexes.get(kj))) {
                                            correctRec = false;
                                            break;
                                        }
                                    }
                                } //end if is numeric

                            }

                        } //end for attr

                        if (correctRec) {
                            theVal = dsToCheck.get(k).value(numericalOnlyIndexes.get(j));
                            break;
                        }

                    } //end for recs in dstocheck

                    r_iChanged.setValue(j, theVal);

                } //end if missing and numeric
                attrCount++;
            } //end attr loop
        } //end missing loop

        /* Step 5: Combine records to form a completed data set D'F without any 
                   missing values. */
        Instances finalDs = new Instances(m_dataset,0);
        int mCount = 0;
        for(int i = 0; i < m_dataset.size(); i++) {
            if(m_dataset.get(i).hasMissingValue()) {
                finalDs.add(m_datasetMissingCopy.get(mCount));
                mCount++;
            }
            else {
                finalDs.add(m_dataset.get(i));
            }
        }
        //m_datasetComplete.addAll(m_datasetMissing);

        m_j48MinRecordsInLeaf = tmpj48Rec;
        m_minRecordsForEMI = tmpMinEMI;
        
        return finalDs;

    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input instance
     * structure (any instances contained in the object are ignored - only the
     * structure is required).
     * @return true if the outputFormat may be collected immediately
     * @throws Exception if the input format can't be set successfully
     */
    public boolean setInputFormat(Instances instanceInfo)
            throws Exception {

        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        return true;
    }

    /**
     * Merges small rules (as strings) into ones that exceed m_minRecordsForEMI
     *
     * @param ruleList - list of rules to merge
     * @return ruleList, with small rules removed and replaced with larger
     * merged rules
     */
    protected ArrayList<String> mergeSmallRules(ArrayList<String> ruleList) {

        boolean stillMerging = true;
        while (stillMerging) {

            //set up a hashmap to store changes we'll need to make
            TreeMap<Integer, String> changeMap = new TreeMap<>();

            for (int i = 0; i < ruleList.size(); i++) {
                stillMerging = false;
                //grab out the rule itself and the number of records that fall in it
                String thisRule = ruleList.get(i);

                //continue if this rule is the whole dataset
                if (thisRule.equals("all")) {
                    continue;
                }

                String[] ruleSplit = thisRule.split(" ");
                
                int leafSize = 0;
                try {
                    leafSize = Integer.parseInt(ruleSplit[ruleSplit.length - 1].replace("(", ""));
                }
                catch(Exception e) {
                    e.printStackTrace();
                }

                if (leafSize < m_minRecordsForEMI) {
                    stillMerging = true;
                    //get the level of this rule by counting ~ characters
                    int ruleLevel = thisRule.length() - thisRule.replace("~", "").length();

                    //get the rule minus the final
                    if (ruleLevel == 0) {
                        //turn this rule into the whole dataset ("all")
                        changeMap.put(i, "all");
                    } else {
                        //cut off the final part of the rule and add the support values
                        String[] thisRuleComponents = thisRule.split(" ~ ");

                        StringBuilder lastOmittedSB = new StringBuilder("");
                        for (int j = 0; j < thisRuleComponents.length - 1; j++) {
                            if (j != 0) {
                                lastOmittedSB.append(" ~ ");
                            }
                            lastOmittedSB.append(thisRuleComponents[j]);
                        }

                        //add up the supports for all rules that start with this 
                        int support = 0;
                        for (int j = 0; j < ruleList.size(); j++) {
                            if (ruleList.get(j).indexOf(lastOmittedSB.toString()) == 0) {
                                String supportStr = ruleList.get(j).substring(ruleList.get(j).lastIndexOf("(") + 1);
                                support += Integer.parseInt(supportStr);
                            }
                        }

                        lastOmittedSB.append(" (").append(support);
                        changeMap.put(i, lastOmittedSB.toString());

                    } //end if rule level

                } //end if leaf is small

            } //end for loop of leaves

            //iterate through and actually make the changes
            for (Map.Entry<Integer, String> entry : changeMap.descendingMap().entrySet()) {

                //ruleList.set(entry.getKey(), entry.getValue());
                ruleList.remove((int) entry.getKey());
                ruleList.add(entry.getValue());

            }

            //remove duplicate merged rules in reverse size order so that the most granular
            //rule is the one which is picked when assigning records to leaves
            int start = ruleList.size() - changeMap.size();
            ArrayList<Integer> toRemove = new ArrayList<>();
            for (int i = start; i < ruleList.size() - 1; i++) {
                String ruleToCheck;
                if (ruleList.get(i).equals("all")) {
                    ruleToCheck = "all";
                } else {
                    ruleToCheck = ruleList.get(i).substring(0, ruleList.get(i).lastIndexOf("("));
                }
                for (int j = i + 1; j < ruleList.size(); j++) {
                    if (ruleList.get(j).equals("all") && !ruleToCheck.equals("all")) {
                        continue;
                    }
                    if ((ruleToCheck.equals("all") && ruleList.get(j).equals("all"))
                            || (ruleToCheck.equals(ruleList.get(j).substring(0,
                                    ruleList.get(j).lastIndexOf("("))))) {
                        toRemove.add(i);
                        break;
                    }
                }
            }
            for (int i = toRemove.size() - 1; i >= 0; i--) {
                ruleList.remove((int) toRemove.get(i));
            }//end duplicate removal
            //ensure reverse size order 
            TreeMap<Integer, ArrayList<String>> ordering = new TreeMap<Integer, ArrayList<String>>();
            for (int i = ruleList.size() - 1; i >= start; i--) {
                String thisRule = ruleList.get(i);
                int ruleLevel;
                if (thisRule.equals("all")) {
                    ruleLevel = -1;
                } else {
                    ruleLevel = thisRule.length() - thisRule.replace("~", "").length();
                }

                if (!ordering.containsKey(ruleLevel)) {
                    ordering.put(ruleLevel, new ArrayList<>());
                }

                ordering.get(ruleLevel).add(thisRule);
                ruleList.remove(i);

            }
            for (Map.Entry<Integer, ArrayList<String>> entry : ordering.descendingMap().entrySet()) {
                for (int i = 0; i < entry.getValue().size(); i++) {
                    ruleList.add(entry.getValue().get(i));
                }
            }

        } //end while stillMerging
        return ruleList;

    }

    /**
     * Compute closest mean for a given record
     *
     * @param rec - The record to check against the means
     * @param meanList - Previously computed list of mean records of leaves
     * @return index of meanlist for closest
     */
    public int closestMean(Instance rec, ArrayList<Instance> meanList) {

        int closest = -1;
        double closestDistance = Double.MAX_VALUE;
        for (int i = 0; i < meanList.size(); i++) {

            double dist = 0;

            for (int j = 0; j < rec.numAttributes(); j++) {

                if (rec.isMissing(j)) {
                    continue;
                }

                if (rec.attribute(j).isNumeric()) {
                    double normalised = (rec.value(j) - m_dataset.attributeStats(j).numericStats.min)
                            / (m_dataset.attributeStats(j).numericStats.min - m_dataset.attributeStats(j).numericStats.max);
                    dist += Math.pow(normalised - meanList.get(i).value(j), 2);
                } else if (rec.value(j) != meanList.get(i).value(j)) {
                    dist += 1.0;
                }

            }

            if (dist < closestDistance) {
                closestDistance = dist;
                closest = i;
            }
        }

        return closest;

    }

    /**
     * Returns minimum number of records in a leaf for J48.
     *
     * @return minimum number of records in a leaf for J48.
     */
    public int getJ48MinRecordsInLeaf() {
        return m_j48MinRecordsInLeaf;
    }

    /**
     * Set minimum number of records in a leaf for J48.
     *
     * @param j48MinRecordsInLeaf - new minimum number of records in a leaf for
     * J48
     */
    public void setJ48MinRecordsInLeaf(int j48MinRecordsInLeaf) {
        m_j48MinRecordsInLeaf = j48MinRecordsInLeaf;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String j48MinRecordsInLeafTipText() {
        return "Minimum number of records in a J48 tree leaf. A negative value will default to (number of numeric attributes in dataset + 2).";
    }

    /**
     * Returns confidence factor for J48.
     *
     * @return confidence factor for J48.
     */
    public float getJ48ConfidenceFactor() {
        return m_j48ConfidenceFactor;
    }

    /**
     * Set confidence factor for J48.
     *
     * @param c - new confidence factor for J48
     */
    public void setJ48ConfidenceFactor(float c) {
        m_j48ConfidenceFactor = c;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String j48ConfidenceFactorTipText() {
        return "Confidence factor for J48.";
    }

    /**
     * Returns minimum number of categories for discretization of numeric
     * attributes.
     *
     * @return confidence factor for J48.
     */
    public int getMinCategoriesForDiscretization() {
        return m_minCategoriesForDiscretization;
    }

    /**
     * Set minimum number of categories for discretization of numeric
     * attributes.
     *
     * @param minCategoriesForDiscretization
     * @throws java.lang.Exception if minimum # categories is less than 2
     */
    public void setMinCategoriesForDiscretization(int minCategoriesForDiscretization)
            throws Exception {
        if (minCategoriesForDiscretization < 2) {
            throw new Exception("Minimum categories for discretization must be >= 2");
        }
        m_minCategoriesForDiscretization = minCategoriesForDiscretization;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String minCategoriesForDiscretizationTipText() {
        return "Minimum number of categories for discretization.";
    }

    /**
     * Returns minimum # records for EMI to run.
     *
     * @return minimum # records for EMI to run.
     */
    public int getMinRecordsForEMI() {
        return m_minRecordsForEMI;
    }

    /**
     * Set minimum # records for EMI to run.
     *
     * @param minRecordsForEMI - minimum # records for EMI to run
     */
    public void setMinRecordsForEMI(int minRecordsForEMI) {
        m_minRecordsForEMI = minRecordsForEMI;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String minRecordsForEMITipText() {
        return "Minimum records in a leaf for EMI to be able to run. A negative value will default to (number of numeric attributes in dataset + 2).";
    }

    /**
     * Returns max # iterations for EMI.
     *
     * @return max # iterations for EMI.
     */
    public int getEmiNumIterations() {
        return m_emiNumIterations;
    }

    /**
     * Set max number of iterations for EMI.
     *
     * @param emiNumIterations - max number of iterations
     */
    public void setEmiNumIterations(int emiNumIterations) {
        m_emiNumIterations = emiNumIterations;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String emiNumIterationsTipText() {
        return "Iterations for EMI. A negative value will be set to Integer.MAX_VALUE";
    }

    /**
     * Returns log likelihood threshold for terminating EMI.
     *
     * @return log likelihood threshold for terminating EMI.
     */
    public double getEmiLogLikelihoodThreshold() {
        return m_emiLogLikelihoodThreshold;
    }

    /**
     * Set log likelihood threshold for terminating EMI.
     *
     * @param emiLogLikelihoodThreshold - log likelihood threshold for
     * terminating EMI
     */
    public void setEmiLogLikelihoodThreshold(double emiLogLikelihoodThreshold) {
        m_emiLogLikelihoodThreshold = emiLogLikelihoodThreshold;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String emiLogLikelihoodThresholdTipText() {
        return "Log likelihood threshold for terminating EMI.";
    }

 
    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start -->
     * Valid options are:
     * <p/>
     * <
     * pre> -D
     * minCategoriesForDiscretization - Minimum number of categories for discretization</pre>
     *
     * <pre> -N
     * j48MinRecordsInLeaf - Minimum number of records in a J48 tree leaf. A negative value will default to (number of numeric attributes in dataset + 2).</pre>
     *
     * <pre> -F
     * j48ConfidenceFactor - Confidence factor for J48</pre>
     *
     * <pre> -E
     * minRecordsForEMI - Minimum records in a leaf for EMI to be able to run. A negative value will default to (number of numeric attributes in dataset + 2)</pre>
     *
     * <pre> -I
     * emiNumIterations - Iterations for EMI. A negative value will be set to Integer.MAX_VALUE</pre>
     *
     * <pre> -L
     * emiLogLikelihoodThreshold - Log likelihood threshold for terminating EMI.</pre>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String optionString;

        // set min # categories for discretization
        optionString = Utils.getOption('D', options);
        if (optionString.length() != 0) {
            setMinCategoriesForDiscretization(Integer.parseInt(optionString));
        }

        // set min # records in J48 tree leaves
        optionString = Utils.getOption('N', options);
        if (optionString.length() != 0) {
            setJ48MinRecordsInLeaf(Integer.parseInt(optionString));
        }

        // set confidence factor for J48
        optionString = Utils.getOption('F', options);
        if (optionString.length() != 0) {
            setJ48ConfidenceFactor(Float.parseFloat(optionString));
        }

        // set min records for emi to run
        optionString = Utils.getOption('E', options);
        if (optionString.length() != 0) {
            setMinRecordsForEMI(Integer.parseInt(optionString));
        }

        // set # iterations for EMI
        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            setEmiNumIterations(Integer.parseInt(optionString));
        }

        // set Log likelihood threshold for terminating EMI
        optionString = Utils.getOption('L', options);
        if (optionString.length() != 0) {
            setEmiLogLikelihoodThreshold(Double.parseDouble(optionString));
        }

    }

    /**
     * Gets the current settings of EMImputation
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {

        Vector<String> result = new Vector<String>();

        result.add("-D");
        result.add("" + getMinCategoriesForDiscretization());

        result.add("-N");
        result.add("" + getJ48MinRecordsInLeaf());

        result.add("-F");
        result.add("" + getJ48ConfidenceFactor());

        result.add("-E");
        result.add("" + getMinRecordsForEMI());

        result.add("-I");
        result.add("" + getEmiNumIterations());

        result.add("-L");
        result.add("" + getEmiLogLikelihoodThreshold());

        return result.toArray(new String[result.size()]);

    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Rahman, M. G.,  & Islam, M. Z.");
        result.setValue(TechnicalInformation.Field.YEAR, "2013");
        result.setValue(TechnicalInformation.Field.TITLE, "Using Decision Trees and Decision Forests by Splitting and Merging Records: Two Novel Techniques");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Knowledge-Based Systems");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Elsevier");
        result.setValue(TechnicalInformation.Field.VOLUME, "53");
        result.setValue(TechnicalInformation.Field.PAGES, "51-65");
        result.setValue(TechnicalInformation.Field.ISSN, "0950-7051");
        result.setValue(TechnicalInformation.Field.URL, "http://www.sciencedirect.com/science/article/pii/S0950705113002591");

        return result;

    }

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {

        return "Class that implements the DMI imputation algorithm for imputing "
                + "missing values in a dataset. DMI splits the dataset into horizontal "
                + "segments using a C4.5 (J48) decision tree in order to increase "
                + "the correlation between attributes for EMI. EMI is performed "
                + "to impute missing numerical attribute values and mean/mode "
                + "(within a leaf) imputation is used to perform missing categorical "
                + "attribute values. Uses Amri Napolitano's EMI implementation for Weka.\n"
                + "\n"
                + "DMI specification from:\n"
                + "Rahman, M. G., and Islam, M. Z. (2013): Missing Value Imputation "
                + "Using Decision Trees and Decision Forests by Splitting and Merging "
                + "Records: Two Novel Techniques, Knowledge-Based Systems, Vol. 53, "
                + "pp. 51 - 65, ISSN 0950-7051, DOI information: 10.1016/j.knosys.2013.08.023, "
                + "Available at http://www.sciencedirect.com/science/article/pii/S0950705113002591\n"
                + "\n"
                + "Changes:\n"
                + "-Leaves that are too small to run EMI are replaced by the nearest "
                + "node above them in a tree. We call this process \"merging\".\n"
                + "-Records that cannot be assigned to any leaves in a tree for "
                + "imputing an attribute will be assigned to the leaf with the "
                + "closest centroid.\n"
                + "-If there are too few records in the whole dataset to ever run"
                + " EMI (i.e. number of records < (number of numeric attributes in dataset + 2)) no "
                + "merging takes place.\n\n"
                + "Valid options are: \n"
                + " -D\n"
                + "minCategoriesForDiscretization - Minimum number of categories for discretization\n\n"
                + "-N\n"
                + "j48MinRecordsInLeaf - Minimum number of records in a J48 tree leaf. A negative value will default to (number of numeric attributes in dataset + 2).\n\n"
                + "-F\n"
                + "j48ConfidenceFactor - Confidence factor for J48\n\n"
                + "-E\n"
                + "m_minRecordsForEMI - Minimum records in a leaf for EMI to be able to run. A negative value will default to (number of numeric attributes in dataset + 2)\n\n"
                + "-I\n"
                + "m_emiNumIterations - Iterations for EMI. A negative value will be set to Integer.MAX_VALUE\n\n"
                + "-L\n"
                + "emiLogLikelihoodThreshold - Log likelihood threshold for terminating EMI.\n\n"
                + "For more information see:\n" + getTechnicalInformation().toString();

    }
    
    boolean allNumericMissing(Instance r, Object[] numeric) {
        
        for(Object n : numeric) {
            if(!r.isMissing((Integer)n)) {
                return false;
            }
        }
        return true;
        
    }

}
