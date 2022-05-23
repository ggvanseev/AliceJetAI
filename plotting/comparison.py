import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import branch_names as na


def hist_comparison(
    anomaly: np,
    normal: np,
    feature: str,
    jet_info=None,
    n_bins=50,
    xlim=None,
    control: dict and ak = None,
):
    """
    control needs two elements: "pythia_normal",  "pythia_anomaly" and is only meant for looking at jewel
    """
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(1.36 * 8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Ensure same bin-size.
    dist_combined, bins = np.histogram(
        np.hstack((normal, anomaly)), bins=n_bins, density=True
    )

    # set the spacing between subplots.
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.99, wspace=0.4, hspace=0)

    # plot top picture, showing
    ax[0].hist(
        normal,
        bins,
        label=f"normal, average = {np.round(np.mean(normal),2)} +/- {np.round(np.std(normal),2)}",
        density=True,
        color="blue",
        histtype="step",
    )

    percentage_anomalous = np.round(
        len(anomaly) / (len(anomaly) + len(normal)) * 100, 2
    )

    ax[0].hist(
        anomaly,
        bins,
        label=f"anomaly ({percentage_anomalous}%), average = {np.round(np.mean(anomaly),2)} +/- {np.round(np.std(anomaly),2)}",
        density=True,
        color="red",
        histtype="step",
    )

    ax[0].set_ylabel(na.axis_n_devide_njets_bandwith)

    # plot ratio
    dist_normal = np.histogram(normal, bins, density=True)[0]
    dist_anomaly = np.histogram(anomaly, bins, density=True)[0]

    ratio_normal = dist_normal / dist_normal
    ratio_anomaly = dist_anomaly / dist_normal

    ax[1].plot(bins[1:], ratio_anomaly, color="red")
    ax[1].plot(bins[1:], ratio_normal, color="blue")

    if control:
        dist_pythia_normal = np.histogram(
            ak.to_numpy(control["pythia_normal"][feature]), bins, density=True
        )[0]
        dist_pythia_anomaly = np.histogram(
            ak.to_numpy(control["pythia_anomaly"][feature]), bins, density=True
        )[0]
        ratio_corrected = (dist_anomaly / dist_pythia_anomaly) / (
            dist_normal / dist_pythia_normal
        )
        ax[1].plot(
            bins[1:],
            ratio_corrected,
            color="green",
        )
        # plot only for  show label
        ax[0].plot([0, 0.0001], [0, 0.0001], color="green", label="Corrected ratio")
        # defintion corrected ratio: [(Anomaly(jewel)/Anomaly(pythia)]/[(Normal(jewel)/Normal(pythia)]

    ax[1].set_xlabel(feature)
    ax[1].set_ylabel("ratio")

    ratio_anomaly_max = ratio_anomaly[~np.isinf(ratio_anomaly)]
    ratio_normal_max = ratio_normal[~np.isinf(ratio_normal)]

    max = np.nanmax(ratio_anomaly_max)
    min = np.nanmin(ratio_anomaly_max)

    ax[1].set_ylim([min, max])
    if xlim:
        ax[1].set_xlim([0, xlim])
    else:
        ax[1].set_xlim([0, bins[-1] + bins[1]])
    ax[0].legend()

    # move spines
    ax[0].spines["left"].set_position(("data", 0.0))
    ax[1].spines["left"].set_position(("data", 0.0))

    return fig


def hist_comparison_first_entries(
    anomaly: ak,
    normal: ak,
    feature: str,
    jet_info=None,
    n_bins=50,
    save_flag=False,
    job_id: str = None,
    num=None,
    xlim=None,
):
    fig = hist_comparison(
        anomaly=ak.to_numpy(ak.firsts(anomaly[feature])),
        normal=ak.to_numpy(ak.firsts(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
        xlim=None,
    )
    if save_flag:
        origin_stamp = f"job_{job_id}_num_{num}"
        plt.savefig(
            f"output/comperative_hist_first_entries_{origin_stamp}__{feature}_{jet_info}.png"
        )


def hist_comparison_flatten_entries(
    anomaly: ak,
    normal: ak,
    feature: str,
    jet_info=None,
    n_bins=50,
    save_flag=False,
    job_id: str = None,
    num=None,
    xlim=None,
):
    fig = hist_comparison(
        anomaly=ak.to_numpy(ak.flatten(anomaly[feature])),
        normal=ak.to_numpy(ak.flatten(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
        xlim=None,
    )
    if save_flag:
        origin_stamp = f"job_{job_id}_num_{num}"
        plt.savefig(
            f"output/comperative_hist_flatten_entries_{origin_stamp}_{feature}_{jet_info}.png"
        )


def hist_comparison_non_recur(
    anomaly: ak,
    normal: ak,
    feature: str,
    jet_info=None,
    n_bins=50,
    save_flag=False,
    job_id: str = None,
    num=None,
    xlim=None,
    control: dict and ak = None,
):
    fig = hist_comparison(
        anomaly=ak.to_numpy(anomaly[feature]),
        normal=ak.to_numpy(normal[feature]),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
        control=control,
    )
    if save_flag:
        origin_stamp = f"job_{job_id}_num_{num}"
        plt.savefig(
            f"output/comperative_hist_non_recur_{origin_stamp}__{feature}_{jet_info}.png"
        )


def hist_comparison_jewel_vs_pythia(
    pythia: np,
    jewel: np,
    feature: str,
    jet_info=None,
    n_bins=50,
    xlim: float = None,
):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(1.36 * 8, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Ensure same bin-size.
    dist_combined, bins = np.histogram(
        np.hstack((jewel, pythia)), bins=n_bins, density=True
    )

    # set the spacing between subplots.
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.99, wspace=0.4, hspace=0)

    # plot top picture, showing
    ax[0].hist(
        pythia,
        bins,
        label=f"pythia, average = {np.round(np.mean(pythia),2)} +/- {np.round(np.std(pythia),2)}",
        density=True,
        color="blue",
        histtype="step",
    )

    ax[0].hist(
        jewel,
        bins,
        label=f"jewel, average = {np.round(np.mean(jewel),2)} +/- {np.round(np.std(jewel),2)}",
        density=True,
        color="red",
        histtype="step",
    )

    ax[0].set_ylabel(na.axis_n_devide_njets_bandwith)

    # plot ratio
    dist_anomaly = np.histogram(jewel, bins, density=True)[0]
    dist_normal = np.histogram(pythia, bins, density=True)[0]

    ratio_normal = dist_normal / dist_normal
    ratio_anomaly = dist_anomaly / dist_normal

    ax[1].plot(bins[1:], ratio_anomaly, color="red")
    ax[1].plot(bins[1:], ratio_normal, color="blue")

    ax[1].set_xlabel(feature)
    ax[1].set_ylabel("ratio")

    ratio_anomaly_max = ratio_anomaly[~np.isinf(ratio_anomaly)]
    ratio_normal_max = ratio_normal[~np.isinf(ratio_normal)]

    max = np.nanmax(ratio_anomaly_max)
    min = np.nanmin(ratio_anomaly_max)

    ax[1].set_ylim([min, max])
    if xlim:
        ax[1].set_xlim([0, xlim])
    else:
        ax[1].set_xlim([0, bins[-1] + bins[1]])
    ax[0].legend()

    # move spines
    ax[0].spines["left"].set_position(("data", 0.0))
    ax[1].spines["left"].set_position(("data", 0.0))

    return fig


def hist_comparison_non_recur_jewel_vs_pythia(
    pythia: ak,
    jewel: ak,
    feature: str,
    jet_info=None,
    n_bins=50,
    save_flag=False,
    job_id: str = None,
    num=None,
    xlim: float = None,
):
    fig = hist_comparison_jewel_vs_pythia(
        pythia=ak.to_numpy(pythia[feature]),
        jewel=ak.to_numpy(jewel[feature]),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
        xlim=xlim,
    )
    if save_flag:
        origin_stamp = f"job_{job_id}_num_{num}"
        plt.savefig(
            f"output/comperative_hist_non_recur_jewel_vs_pythia_{origin_stamp}__{feature}_{jet_info}.png"
        )
