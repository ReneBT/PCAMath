### Manual adjustment of the scaling factor
################# DO NOT DELETE ############################
xview = range(801)
iPC = 12
adj = [
    0,
    1,
    0.993,
    1.001,
    0.995,
    1.002,
    0.995,
    1.0015,
    1.0005,
    1.0002,
    0.9995,
    1.00015,
    1,
]
a = plt.subplot2grid((1, 1), (0, 0), colspan=1)
a.plot(FAsim["FAXcal"][[0, 0]][0,][[xview[0], xview[-0]]], [0, 0], "--")
a.plot(
    FAsim["FAXcal"][[0, 0]][0,][xview],
    PCAFullCovNMC.nCon[xview, iPC]
    - PCAFullCovNMC.cCon[xview, iPC] * (PCAFullCovNMC.optSF[iPC] * adj[iPC]),
    "b",
    linewidth=1,
)
a.plot(
    FAsim["FAXcal"][[0, 0]][0,][xview],
    PCAFullCovNMC.pCon[xview, iPC]
    - PCAFullCovNMC.cCon[xview, iPC] * (PCAFullCovNMC.optSF[iPC] * adj[iPC]),
    "y",
    linewidth=1,
)
# a.plot( FAsim['FAXcal'][[0,0]][0,][xview]  , PCAFullCovNMC.cConS[xview,iPC] , 'g', linewidth=0.5)#*smsf[iPC]
a.plot(
    FAsim["FAXcal"][[0, 0]][0,][xview],
    PCAFullCovNMC.mConS[xview, iPC],
    "k",
    linewidth=0.5,
)  # *smsf[iPC]
txtpos = [np.mean(a.get_xlim()), a.get_ylim()[1] * 0.9]
a.annotate(
    "PC " + str(iPC + 1),
    xy=(txtpos),
    xytext=(txtpos),
    textcoords="data",
    fontsize=8,
    horizontalalignment="left",
)

plt.show()

PCAFullCovNMC.optSFm = PCAFullCovNMC.optSF * adj
plt.figure(figsize=(8, 8))
figsmsfm = plt.plot(
    range(2, np.shape(PCAFullCovNMC.optSFm)[0] + 1), PCAFullCovNMC.optSFm[1:], "."
)
drp = np.add(np.nonzero((PCAFullCovNMC.optSFm[2:] - PCAFullCovNMC.optSFm[1:-1]) < 0), 2)
if np.size(drp) != 0:
    drp2 = drp[0][0]
    plt.plot(drp2 + 1, PCAFullCovNMC.optSFm[drp2], "or")
    plt.plot([2, nPC], [PCAFullCovNMC.optSFm[drp2], PCAFullCovNMC.optSFm[drp2]], "--")
plt.savefig("img\lpniCommonSignalScalingFactorsManuallyAdjusted.png", dpi=300)
plt.close()

## NEW CELL ##
## plot each zoomed in region for checking subtraction is OK across each part of spectrum - re run this cell with # a different xview (or feel free to create an array to range over)
xview = range(
    400, 500
)  # range(range(625,695) #range(700,801) #range(200,351) #range(0,151) #
figlpniSA, axlpniSA = plt.subplots(
    1, 6, figsize=(8, 8)
)  # Extra columns to match spacing in
for iPC in range(1, nPC):
    iFig = iPC % 6 - 1  # modulo - determine where within block the current PC is
    if iFig == -1:
        iFig = 5  # if no remainder then it is the last in the plot
    axlpniSA[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
    axlpniSA[iFig].plot(
        FAsim["FAXcal"][[0, 0]][0,][[xview[0], xview[-0]]], [0, 0], "--"
    )
    axlpniSA[iFig].plot(
        FAsim["FAXcal"][[0, 0]][0,][xview],
        PCAFullCovNMC.nCon[xview, iPC]
        - PCAFullCovNMC.cCon[xview, iPC] * (PCAFullCovNMC.optSF[iPC] * adj[iPC]),
        "b",
        linewidth=1,
    )
    axlpniSA[iFig].plot(
        FAsim["FAXcal"][[0, 0]][0,][xview],
        PCAFullCovNMC.pCon[xview, iPC]
        - PCAFullCovNMC.cCon[xview, iPC] * (PCAFullCovNMC.optSF[iPC] * adj[iPC]),
        "y",
        linewidth=1,
    )
    axlpniSA[iFig].plot(
        FAsim["FAXcal"][[0, 0]][0,][xview],
        PCAFullCovNMC.cConS[xview, iPC],
        "g",
        linewidth=0.5,
    )  # *smsf[iPC]
    axlpniSA[iFig].plot(
        FAsim["FAXcal"][[0, 0]][0,][xview],
        PCAFullCovNMC.mConS[xview, iPC],
        "k",
        linewidth=0.5,
    )  # *smsf[iPC]
    txtpos = [np.mean(axlpniSA[iFig].get_xlim()), axlpniSA[iFig].get_ylim()[1] * 0.9]
    axlpniSA[iFig].annotate(
        "PC " + str(iPC + 1),
        xy=(txtpos),
        xytext=(txtpos),
        textcoords="data",
        fontsize=8,
        horizontalalignment="left",
    )
    if iFig == 5:
        for iax in range(np.shape(axlpniSA)[0]):
            axlpniSA[iax].axis("off")
        figlpniSA.savefig(
            "img\lpniAdjustedSubLEigenvectorEqn_"
            + str(iPC - 4)
            + "_"
            + str(iPC + 1)
            + " "
            + str(xview[0])
            + "_"
            + str(xview[-1])
            + ".png",
            dpi=300,
        )
        plt.close()
        figlpniSA, axlpniSA = plt.subplots(
            1, 6, figsize=(8, 8)
        )  # Extra columns to match spacing in
