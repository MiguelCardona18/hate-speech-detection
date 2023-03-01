# Python program to convert .tsv file to .csv file
# importing re library
import re
  
# reading given tsv file
with open("offenseval-tr-labela-v1.tsv", 'r') as myfile:  
  with open("test-tr-labels.csv", 'w') as csv_file:
    for line in myfile:
        
      # Replace every tab with comma
      fileContent = re.sub("\t", ",", line)
        
      # Writing into csv file
      csv_file.write(fileContent)
  
# output
print("Successfully made csv file")