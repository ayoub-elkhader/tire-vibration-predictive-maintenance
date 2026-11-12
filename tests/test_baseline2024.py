import numpy as np

from tire_pdm import baseline2024 as b


def test_method1_matches_2024_csv_first_rows():
    d = b.method1()
    # first two rows of Amplitude_Frequency_Data.csv (2024)
    assert np.allclose(d.iloc[0, 1:], [-1.93820026016113, -3.52182518111363, -2.498774732166], atol=1e-12)
    assert np.allclose(d.iloc[1, 1:], [-1.93818679134398, -3.52180303849061, -2.49875798203977], atol=1e-12)


def test_method2_matches_2024_csv_first_row():
    d = b.method2()
    assert np.allclose(d.iloc[0], [0.1, 0.1, 0.106793871837085, 0.149462624689934, 0.159616923855712,
                                   0.237877155268918, 0.254038224327592], rtol=1e-12)


def test_training_table_has_2024_layout():
    t = b.training_table(b.method1())
    assert list(t.columns) == ["Frequence", "Magnitude", "Panne"]
    assert len(t) == 3000 and (t.Panne == 1).sum() == 2000
