plates = [1,2,3,4,5]
ratios = {
    1:'1:1',
    2:'1:3',
    3:'1:5',
    4:'3:1',
    5:'5:1'
}
for plate_num in plates:
    plate1 = pd.read_csv(f"https://raw.githubusercontent.com/SanatRao13/engr100-finalproj/main/ENGR100_Plate{plate_num}Data.csv")
    plate1 = plate1.drop(columns=['12', 'Unnamed: 0'])

    ser = []
    for col in plate1.columns:
        ser.append(plate1[col].mean())
    ser = pd.Series(ser)

    plate1_blood = plate1.iloc[:3].reset_index(drop=True)
    plate1_mucus = plate1.iloc[4:7].reset_index(drop=True)

    plate1_blood.loc[4] = plate1_blood.loc[:3].mean()
    plate1_mucus.loc[4] = plate1_mucus.loc[:3].mean()

    plate1_blood = plate1_blood.reset_index(drop=True)
    plate1_mucus = plate1_mucus.reset_index(drop=True)

    plate1_blood = plate1_blood.iloc[:,:10]
    plate1_mucus = plate1_mucus.iloc[:,:10]

    plate1_blood = dye_mass(plate1_blood)*0.2
    plate1_mucus = dye_mass(plate1_mucus)*0.2

    print("\n\n")

    cr_p1b = plate1_blood.astype(float).cumsum(axis=1)
    ncr_p1b = cr_p1b.div(cr_p1b.max(axis=1), axis=0)
    cr_p1m = plate1_mucus.astype(float).cumsum(axis=1)
    ncr_p1m = cr_p1m.div(cr_p1m.max(axis=1), axis=0)

    times = [10*x for x in range(len(cr_p1b.columns))]
    plt.plot(times, ncr_p1b.iloc[3].T, 'ro-', label=f'Plate {plate_num}, Blood Sim')
    plt.plot(times, ncr_p1m.iloc[3].T, 'yo-', label=f'Plate {plate_num}, Mucus Sim')
    plt.title(f"Normalized Cumulative Release Profiles of PVA Beads\nin Blood versus Mucus Environments ({ratios[plate_num]} of PVA:Na-Alginate)")
    plt.xticks(np.arange(0, 95, 5))
    plt.yticks(np.arange(0, max(max(ncr_p1b.iloc[3]), max(ncr_p1m.iloc[3]))*21/20, max(max(ncr_p1b.iloc[3]), max(ncr_p1m.iloc[3]))/20))
    plt.legend()
    plt.grid()
    plt.show()
