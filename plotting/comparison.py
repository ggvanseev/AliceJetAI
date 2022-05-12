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
):
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
        label="normal",
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
        label=f"anomaly ({percentage_anomalous}%)",
        density=True,
        color="red",
        histtype="step",
    )

    ax[0].hist(
        np.hstack((normal, anomaly)),
        bins,
        label="combined",
        density=True,
        color="black",
        histtype="step",
    )
    ax[0].set_ylabel(na.axis_n_devide_njets_bandwith)

    # plot ratio
    dist_normal = np.histogram(normal, bins, density=True)[0]
    dist_anomaly = np.histogram(anomaly, bins, density=True)[0]

    ratio_combined = dist_combined / dist_combined
    ratio_normal = dist_normal / dist_combined
    ratio_anomaly = dist_anomaly / dist_combined

    ax[1].plot(bins[1:], ratio_combined, color="black")
    ax[1].plot(bins[1:], ratio_anomaly, color="red")
    ax[1].plot(bins[1:], ratio_normal, color="blue")

    ax[1].set_xlabel(feature)
    ax[1].set_ylabel("ratio")

    max = (
        np.nanmax(ratio_anomaly[1:])
        if np.nanmax(ratio_anomaly[1:]) > np.nanmax(ratio_normal[1:])
        else np.nanmax(ratio_normal[1:])
    )

    ax[1].set_ylim([0, max])
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
):
    fig = hist_comparison(
        anomaly=ak.to_numpy(ak.firsts(anomaly[feature])),
        normal=ak.to_numpy(ak.firsts(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
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
):
    fig = hist_comparison(
        anomaly=ak.to_numpy(ak.flatten(anomaly[feature])),
        normal=ak.to_numpy(ak.flatten(normal[feature])),
        feature=feature,
        jet_info=jet_info,
        n_bins=n_bins,
    )
    if save_flag:
        origin_stamp = f"job_{job_id}_num_{num}"
        plt.savefig(
            f"output/comperative_hist_flatten_entries_{origin_stamp}_{feature}_{jet_info}.png"
        )
