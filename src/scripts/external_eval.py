from lxml import etree as ET
from xml.sax.saxutils import unescape
import subprocess, os

def evaluate_alignment(trainer, test_loader, script_file, gold_ali_file, tmp_file_name='/tmp/output'):
    assignments = []
    for idx, p in enumerate(test_loader.dataset):
        pr = trainer.predict_assignment(p)[1]
        assignments.append((idx +1 , pr))

    sentences = ET.Element("sentences")
    for al in assignments:
        sentence = ET.SubElement(sentences, "sentence")
        sentence.set("id", str(al[0]))
        sentence.set("status", "")
        alignment = ET.SubElement(sentence, "alignment")
        #Dummy fill the rest of the columns, as we are interested only in alignment
        s = "\n".join([ e + " // EQUI // 5 // for the Philippines" for e in al[1]])
        alignment.text = "\n" + s + "\n"

    xml_data = unescape(ET.tostring(sentences))
    with open(tmp_file_name, "w") as fp:
        fp.write(xml_data)

    fixed_env = dict(os.environ, LC_CTYPE='en_US.UTF-8', LC_ALL='en_US.UTF-8')
    p = subprocess.Popen(['perl', script_file, gold_ali_file, tmp_file_name], stdout=subprocess.PIPE, env=fixed_env)
    f1_score = float(p.stdout.read().split()[2])

    return f1_score

if __name__ == '__main__':
    script_file = '../datasets/sts_16/test/evalF1.pl'
    gold_ali_file = '../datasets/sts_16/test/STSint.testinput.headlines.wa'