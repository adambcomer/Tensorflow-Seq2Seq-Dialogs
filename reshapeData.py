import numpy as np
import re
from itertools import izip

csv = np.loadtxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/movie_conversations.txt", delimiter=" +++$+++ ", dtype="string", skiprows=0)
linescsv = np.loadtxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/reviewed_dialog.csv", delimiter=" +++$+++ ", dtype="string", skiprows=0, comments="")

lineid = []
linetxt = []
tempdict = []
dictionary = set(["NULL", "GO"])

maxlen = 0

for row in linescsv:
    lines = row
    lines = str(lines).split()
    lines = " ".join(lines)
    # lines = lines.replace(".", "")
    # lines = lines.replace("!", "")
    # lines = lines.replace("?", "")
    # lines = lines.replace("-", "")
    # lines = lines.replace("            ", " ")
    # lines = lines.replace("           ", " ")
    # lines = lines.replace("          ", " ")
    # lines = lines.replace("         ", " ")
    # lines = lines.replace("        ", " ")
    # lines = lines.replace("       ", " ")
    # lines = lines.replace("      ", " ")
    # lines = lines.replace("     ", " ")
    # lines = lines.replace("    ", " ")
    # lines = lines.replace("   ", " ")
    # lines = lines.replace("  ", " ")
    lines = lines.replace(",", " ")
    # lines = lines.replace("/", "")
    # lines = lines.replace("' ", " ")
    # lines = lines.replace(" '", " ")
    # lines = lines.replace(" ' ", " ")
    # lines = lines.replace("\"", "")
    # lines = lines.replace("*", "")
    # lines = lines.replace(";", "")
    # lines = lines.replace(":", "")
    # lines = lines.replace("<u>", "")
    # lines = lines.replace("<b>", "")
    # lines = lines.replace("_", "")
    # lines = lines.replace("]", "")
    # lines = lines.replace("[", "")
    # lines = lines.replace("}", "")
    # lines = lines.replace("{", "")
    # lines = lines.replace(")", "")
    # lines = lines.replace("(", "")
    # lines = lines.replace("@", "")
    # lines = lines.replace("#", "")
    # lines = lines.replace("$", "")
    # lines = lines.replace("%", "")
    # lines = lines.replace("^", "")
    # lines = lines.replace("&", "")
    # lines = lines.replace("+", "")
    # lines = lines.replace("=", "")
    # lines = lines.replace("|", "")
    # lines = lines.replace("\\", "")
    # lines = lines.replace("<", "")
    # lines = lines.replace(">", "")
    #lines = lines.decode('utf8', errors="ignore")
    #lines = [item for item in lines if not item.isdigit()]
    line = ''.join(lines).strip()

    #print(line)

    #lineid.append(row[0])
    #linetxt.append(line)

    words = re.split(" ", line)
    tempdict.extend(words)

    if len(words) > maxlen:
        maxlen = len(words)
        #print(words)

dictionary = dictionary.union(tempdict)
print("Number of words in dictionary: " + str(len(dictionary)))

print("Max Length: " + str(maxlen))

linesdict = dict(izip(lineid, linetxt))
print("Number of lines: " + str(linescsv.shape[0]))
newcsv = []

# for row in csv:
#     lines = list(row[3])
#     lines.remove('[')
#     lines.remove(']')
#     lines = [item for item in lines if item != "'"]
#     lines = re.split(", ", ''.join(lines))
#     for i in range(len(lines) - 1):
#         newcsv.append(np.array([linesdict[lines[i]], linesdict[lines[i + 1]]]))

#np.savetxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/dialogs.csv", np.array(newcsv), delimiter=",", fmt="%s,%s")
np.savetxt("/Users/adamcomer/PycharmProjects/TensorflowSeq/dictionary.csv", np.array(list(dictionary)), delimiter=",", fmt="%s")
print("Dialog CSV is saved")