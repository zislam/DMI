/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/** 
 <!-- globalinfo-start -->
 * Parent class for imputation techniques based on using a decision tree algorithm
 * to put the dataset into horizontal segments.
 * <p/>
 <!-- globalinfo-end -->
 *
 * @author Michael Furner 
 */
public abstract class DecisionTreeImputation extends SimpleBatchFilter implements UnsupervisedFilter {

    /** Dataset to be imputed */
    protected Instances m_dataset;
    
    /** Subset of dataset only containing records with missing values */
    protected Instances m_datasetMissing;
    
    /** Subset of dataset only containing complete records */
    protected Instances m_datasetComplete;

    /**
     * Separates complete records and records with missing values from m_dataset.
     * @throws Exception 
     */
    protected void separateMissingRecords() throws Exception {

        m_datasetMissing = new Instances(m_dataset);
        m_datasetComplete = new Instances(m_dataset);

        //initialise the filters that will be used to create the two subsets
        RemoveWithValues removeMissing = new RemoveWithValues();
        removeMissing.setInputFormat(m_dataset);
        removeMissing.setMatchMissingValues(true);

        RemoveWithValues removeComplete = new RemoveWithValues();
        removeComplete.setInputFormat(m_dataset);
        removeComplete.setMatchMissingValues(false);
        removeComplete.setInvertSelection(false);

        for (int j = 0; j < m_dataset.numAttributes(); j++) {

            int attIndex = j + 1;
            removeMissing.setAttributeIndex("" + attIndex);
            removeComplete.setAttributeIndex("" + attIndex);

            //to remove the missing values only, we have to set the
            //split point and nominal values to nonsense values
            removeMissing.setNominalIndices("");
            removeComplete.setNominalIndices("first-last");
            removeMissing.setSplitPoint(Double.NEGATIVE_INFINITY);
            removeComplete.setSplitPoint(Double.POSITIVE_INFINITY);
            removeMissing.setModifyHeader(false);
            removeComplete.setModifyHeader(false);

            removeMissing.setInputFormat(m_dataset);
            removeComplete.setInputFormat(m_dataset);

            m_datasetComplete = Filter.useFilter(m_datasetComplete, removeMissing);

        } //end attr loop

        for (int i = 0; i < m_datasetComplete.numInstances(); i++) {

            Instance toRemove = m_datasetComplete.get(i);

            //jump through m_datasetMissing and remove one at a time
            for (int j = 0; j < m_datasetMissing.numInstances(); j++) {
                Instance toCheck = m_datasetMissing.get(j);

                InstanceComparator ic = new InstanceComparator();
                if (ic.compare(toRemove, toCheck) == 0) {
                    m_datasetMissing.remove(j);
                    break;
                }

            } // end missing dataset 

        } //end remove complete loop

    }

    /**
     * Extracts the rules in simplified single-line string format from 
     * J48 output.
     * @param j48String - String representing decision tree made by J48
     * @param numberOfRules - The number of rules (leaves) in this tree
     * @return 
     */
    protected ArrayList<String> ruleExtractor(String j48String, int numberOfRules) {

       ArrayList<String> ruleList = new ArrayList<String>();
       
       /*if there was no way to build a decision tree on the attribute (e.g. 
         most of the values fall into one class and there's no good way to split)
         then we just indicate that for this attribute we will use the whole 
         dataset */
       if(numberOfRules == 1) {
           ruleList.add("all");
           return ruleList;
       }
       
        String tree = j48String.replace("J48 pruned tree\n------------------\n\n", "");
        String[] lines = tree.split("\n");

        int previousRuleLine = -1;
        for (int i = 0; i < numberOfRules; i++) {

            for (int l = previousRuleLine + 1; l < lines.length; l++) {

                //if this is a leaf
                if (lines[l].contains("(")) {
                    
                    StringBuilder currentRule = new StringBuilder();

                    String thisLine = lines[l];

                    //work out the current level based on indentation
                    int topLevel = 0;
                    if (thisLine.contains("|   ")) {
                        topLevel = (thisLine.lastIndexOf("|   ")) / 4 + 1;
                    }

                    //extract condition HERE 
                    Pattern p = Pattern.compile("[a-zA-Z0-9_-]+ [<=>]+ [a-zA-Z0-9()&_.<=>-]+: [a-zA-Z0-9().\\]\\[&\\\\'_-]+ \\([0-9]+");
                    Matcher m = p.matcher(thisLine);
                    String rulePart = "";
                    if(m.find())
                        rulePart = thisLine.substring(m.start(), m.end());
                    else
                        System.out.println("error");

                    //append to current rule todo
                    currentRule.append(rulePart);

                    for (int k = l; k >= 0; k--) {

                        int currentLevel = 0;
                        if (lines[k].contains("|   ")) {
                            currentLevel = (lines[k].lastIndexOf("|   ")) / 4 + 1;
                        }

                        if (currentLevel < topLevel) {

                            //extract condition HERE todo
                            p = Pattern.compile("[a-zA-Z0-9_-]+ [<=>]+ [a-zA-Z0-9()&_.>=<-]+");
                            m = p.matcher(lines[k]);
                            if(m.find())
                                rulePart = lines[k].substring(m.start(), m.end());
                            else
                                System.out.println("error");


                            //append to current rule todo
                            currentRule.insert(0, " ~ ");
                            currentRule.insert(0, rulePart);

                            topLevel = currentLevel;

                            if (currentLevel < 0) {
                                break;
                            }

                        }

                    }

                    previousRuleLine = l;

                    ruleList.add(currentRule.toString());

                } // end rule condition

            } // end line loop

        } // end rule loop

        return ruleList;

    }

    /**
     * Find the subset of the dataset associated with each rule in a set 
     * using datasetFromRule
     * @param dsToSplit - The dataset to be split up into subsets
     * @param rules - All rules to find corresponding subset for
     * @return
     * @throws Exception 
     */
    protected ArrayList<Instances> datasetsFromRules(Instances dsToSplit, ArrayList<String> rules) {

        ArrayList<Instances> datasetList = new ArrayList<Instances>();

        for (int i = 0; i < rules.size(); i++) {
            datasetList.add(datasetFromRule(dsToSplit, rules.get(i)));
        }

        return datasetList;

    }

    /**
     * Returns subset of records in dsToSplit that correspond with rule
     * @param dsToSplit - record set 
     * @param rule - for splitting
     * @return
     * @throws Exception 
     */
    protected Instances datasetFromRule(Instances dsToSplit, String rule) {

        String[] components = rule.split(" ~ ");
        Instances subDs = new Instances(dsToSplit);
        if(rule.equals("all") ){
            return subDs;
        }

        for (int i = 0; i < components.length; i++) {
            String[] segments = components[i].split(" ");
            String attrName = segments[0];
            String comparator = segments[1];
            String attrValue = segments[2];
            if (i == components.length - 1) { //final component
                attrValue = attrValue.replace(":", "");
            }

            int attrIndex = subDs.attribute(attrName).index() + 1;
            
            if (subDs.attribute(attrName).isNumeric()) {
                ArrayList<Integer> removeList = new ArrayList<>();

                if(comparator.equals("<=")) {
                    for(int j = 0; j < subDs.numInstances(); j++) {
                        if(subDs.get(j).value(attrIndex-1) > Double.parseDouble(attrValue)) {
                            removeList.add(j);
                        }
                    }//end remove recs
                }
                else {
                    for(int j = 0; j < subDs.numInstances(); j++) {
                        if(subDs.get(j).value(attrIndex-1) <= Double.parseDouble(attrValue)) {
                            removeList.add(j);
                        }
                    } //end remove recs                    
                } 
                
                Collections.reverse(removeList);
                for(int j = 0; j < removeList.size(); j++) {
                    subDs.delete(removeList.get(j));
                }
                
            } else { //is categorical

                RemoveWithValues rwv = new RemoveWithValues();
                rwv.setAttributeIndex("" + attrIndex);
                int nomIndex = subDs.attribute(attrName).indexOfValue(attrValue) + 1;
                rwv.setNominalIndices("" + nomIndex);
                rwv.setInvertSelection(true);

                try {
                    rwv.setInputFormat(subDs);
                    subDs = Filter.useFilter(subDs, rwv);
                }
                catch(Exception e ) {
                    e.printStackTrace();
                }

            }

        }

        return subDs;
    }

    /**
     * Gives index of rule that a record will fall into.
     * @param rules - set of rules
     * @param record - to compare against rules
     * @return -1 if none found, 1 otherwise
     */
    int ruleGivenRecord(ArrayList<String> rules, Instance record) {

        //we've got an attribute that couldn't make a good tree
        if(rules.size() == 1) {
            return 0;
        }
        for (int i = 0; i < rules.size(); i++) {
            
            String ruleStr = rules.get(i);
            if(ruleStr.equals("all")) {
                return i;
            }
            String[] components = ruleStr.split(" ~ ");

            boolean thisIsTheRule = true;
            for (String component : components) {
                String[] segments = component.split(" ");
                String attrName = segments[0];
                String comparator = segments[1];
                String attrValue = segments[2];
                if (component.equals(components[components.length - 1])) { //final component
                    attrValue = attrValue.replace(":", "");
                }

                int attrIndex = m_dataset.attribute(attrName).index();
                
                if(m_dataset.attribute(attrIndex).isNumeric()) {
                    if(comparator.equals(">")) {
                        if(record.value(attrIndex) <= Double.parseDouble(attrValue)) {
                            thisIsTheRule = false;
                            break;
                        }
                    } else if(comparator.equals("<=")) {
                        if(record.value(attrIndex) > Double.parseDouble(attrValue)) {
                            thisIsTheRule = false;
                            break;
                        }
                    }
                }
                else { //categorical
                    if((int)(record.value(attrIndex)) != m_dataset.attribute(attrIndex).indexOfValue(attrValue)) {
                        thisIsTheRule = false;
                        break;
                    }
                }
                
            } //end component loop
            
            if(thisIsTheRule) {
                return i;
            }

        }

        return -1;

    }
    

}
