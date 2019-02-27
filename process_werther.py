import numpy as np
import pickle as pkl

def main(pth):
    lines = []
    with open(pth, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.replace('ß', 'ss')
            line = line.replace('--', ' ')
            line = line.replace("'s", ' es')
            line = line.replace('?', ' ? <EOS> ')
            line = line.replace('.', ' . <EOS> ')
            line = line.replace('!', ' ! <EOS> ')
            line = line.replace(',', ' , ')
            line = line.replace("'", '')
            line = line.replace('"', ' " ')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('***', '')
            if line != '':
                lines.append(line)

        werther_clean = [y.lower() for x in lines for y in x.split(' ') if y != '']
        corpus = np.unique(werther_clean)
        one_hot = np.eye(len(corpus))
        labels = {t[0]: t[1] for t in [(a, b) for a, b in zip(corpus, one_hot)]}
        werther_converted = np.array([labels[x] for x in werther_clean])
        eos_label = labels['<eos>']
        werther_sentences = []
        sentence = []
        for w in werther_converted:
            if any(w - eos_label):
                sentence.append(w)
            else:
                sentence.append(w)
                werther_sentences.append(sentence)
                sentence = []

        if True:
            np.save('./data/werther_clean.npy', werther_clean)
            with open('./data/werther_labels.pkl', 'wb') as f:
                pkl.dump(labels, f)
            np.save('./data/werther_converted.npy', werther_converted)
            np.save('./data/werther_sentences.npy', werther_sentences)

if __name__ == '__main__':
    main('./data/werther.txt')