import pandas as pd
from src.fleurs_to_flores import LANGUAGE_MAPPING
from pathlib import Path
from datasets import load_dataset
from src.utils import (
    read_fleurs,
    remove_extra_whitespace,
    remove_punctuation,
)
from src.levenshtein import match_sentences


# Merge fleurs into SIB

# SIB_LANGUAGES = [ "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn", "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Arab", "min_Latn", "mkd_Cyrl", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nqo_Nkoo", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn", "prs_Arab", "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Olck", "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn", ]


def merge(lang):
    lang_folder = LANGUAGE_MAPPING.get(lang, None)
    if lang_folder is None:
        print(f"{lang} not in Fleurs")
        return

    # Dataset({
    #     features: ['index_id', 'category', 'text'],
    #     num_rows: 701
    # })

    try:
        sib_ = {
            k: pd.DataFrame(v) for k, v in load_dataset("Davlan/sib200", lang).items()
        }
        sib = {
            k: pd.DataFrame(v) for k, v in load_dataset("Davlan/sib200", lang).items()
        }
        ok = True
    except:
        ok = False
    if not ok:
        print(f"{lang} not in SIB")

    if ok:
        # Load fleurs dataset
        cwd = Path.cwd()
        paths = ["fleurs", "data", lang_folder]

        fleurs = pd.concat(
            [
                pd.DataFrame.from_records(read_fleurs(t))
                for t in cwd.joinpath(*paths).glob("*.tsv")
            ],
            axis=0,
        )

        fleurs["raw_transcription_normalized"] = fleurs["raw_transcription"].apply(
            lambda sent: remove_punctuation(remove_extra_whitespace(sent))
        )
        fleurs_aggregated = (
            fleurs.groupby("fleurs_id")
            .agg(
                lambda x: tuple(x) if not all(y == x.iloc[0] for y in x) else x.iloc[0]
            )
            .reset_index()
        )
        for split in sib.keys():
            sib[split]["text_normalized"] = sib[split]["text"].apply(
                lambda sent: remove_punctuation(remove_extra_whitespace(sent))
            )
            fleurs_transcriptions = set(
                fleurs_aggregated["raw_transcription_normalized"].tolist()
            )
            sib_transcriptions = set(sib[split]["text_normalized"].tolist())

            exact_matches = fleurs_transcriptions.intersection(sib_transcriptions)
            difference = fleurs_transcriptions.difference(sib_transcriptions)

            # merge into sib so sib order is maintained for later matching with Belebele
            merged = sib[split].merge(
                fleurs_aggregated,
                left_on="text_normalized",
                right_on="raw_transcription_normalized",
                how="inner",
            )
            # NOTE: this cannot be quite true for some languages that have duplicates
            #       we however need those duplicates because they might have different URLs
            # assert len(merged) == len(exact_matches)

            if len(difference) > 0:
                matches: dict[str, str] = match_sentences(
                    list(difference), sib_transcriptions, 0.025
                )
                matches = {k: v for k, v in matches.items() if v is not None}
                fleurs_aggregated["raw_transcription_normalized"] = fleurs_aggregated[
                    "raw_transcription_normalized"
                ].replace(matches)
                merged = sib[split].merge(
                    fleurs_aggregated,
                    left_on="text_normalized",
                    right_on="raw_transcription_normalized",
                    how="inner",
                )
            sib[split] = merged

        print(f"{lang}:")
        for k in sib.keys():
            orig = sib_[k]
            merged = sib[k]
            print(f"    {k}: {len(merged)}/{len(orig)}")


for lang in LANGUAGE_MAPPING:
    merge(lang)
