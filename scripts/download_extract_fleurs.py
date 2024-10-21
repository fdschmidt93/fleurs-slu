from pathlib import Path
import urllib.request
import tarfile
from concurrent.futures import ThreadPoolExecutor

URL = "https://storage.googleapis.com/xtreme_translations/FLEURS/{}.tar.gz"
# fmt: off
# LANGUAGES = sorted( [ "af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in", "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr", "ku_arab_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oci_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru", "rup_bg", "sd_arab_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za", ])
LANGUAGES = sorted( [ "af_za"])
# fmt: on

# Get the current file's directory (where download_and_extract.py is located)
current_file_dir = Path(__file__).resolve().parent

# Set PROJECT_DIR as the parent of the 'scripts' directory (i.e., the project root)
PROJECT_DIR = current_file_dir.parent
output_dir = PROJECT_DIR / "data" / "fleurs"
output_dir.mkdir(exist_ok=True)


def download_and_extract(language: str):
    """Downloads and extracts a file for a specific language."""
    file_url = URL.format(language)
    file_path = output_dir / f"{language}.tar.gz"
    extract_path = output_dir / language

    # Download the file if it doesn't exist
    if not file_path.exists():
        try:
            print(f"Downloading {language}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {language}")
        except Exception as e:
            print(f"Failed to download {language}: {e}")
            return
    else:
        print(f"{file_path} already exists. Skipping download.")

    # Extract the tar.gz file if not already extracted
    if not extract_path.exists():
        try:
            print(f"Extracting {language}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print(f"Extracted {language}")
        except Exception as e:
            print(f"Failed to extract {language}: {e}")
    else:
        print(f"{extract_path} already exists. Skipping extraction.")


def main():
    # Use ThreadPoolExecutor to download in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_and_extract, LANGUAGES)

    print("All downloads completed.")


if __name__ == "__main__":
    main()
