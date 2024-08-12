import neal
from pyqubo import Binary
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# We declare the binary variables which are included in the problem and assume values {0,1}
x1, x2, x3, x4, x5 = Binary("x1"), Binary("x2"), Binary("x3"), Binary("x4"), Binary("x5")


H = 2*x1 + 2*x2 + 3*x3 + 3*x4 + 2*x5 - 2*x1*x2 - 2*x1*x3 - 2*x2*x4 - 2*x3*x4 - 2*x3*x5 - 2*x4*x5
H_maxcut = -H


if __name__ == "__main__":
    list_of_samples = []
    model = H_maxcut.compile()
    bqm = model.to_bqm(index_label=True)
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_reads=50)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    print("Best sample:", best_sample)
    sample_frequencies = Counter([tuple(sample.sample.items()) for sample in decoded_samples])

    total_samples = len(decoded_samples)
    tot_freq = []
    for sample, freq in sample_frequencies.items():
        print("Sample:", dict(sample), "Frequency:", freq)
        tot_freq.append(freq)

    samples = np.arange(0, 4, 1)

    plt.bar(samples, np.divide(np.asarray(tot_freq), total_samples), label="Frequencies")
    plt.xlabel("Samples")
    plt.xticks(samples)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()