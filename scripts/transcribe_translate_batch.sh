#!/usr/bin/bash

# Validate arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 model_name"
    exit 1
fi

MODEL=${1}
valid_models=("whisper" "seamlessm4t" "mms")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL} " ]]; then
    echo "Error: Model must be one of: ${valid_models[*]}"
    exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
project_root="/network/scratch/s/schmidtf/fleurs-slu"
mkdir -p "${project_root}/logs/${MODEL}/${timestamp}"

# for lang in "af_za" "am_et" "ar_eg" "as_in" "ast_es" "az_az" "be_by" "bg_bg" "bn_in" "bs_ba" "ca_es" "ceb_ph" "ckb_iq" "cmn_hans_cn" "cs_cz" "cy_gb" "da_dk" "de_de" "el_gr" "en_us" "es_419" "et_ee" "fa_ir" "ff_sn" "fi_fi" "fil_ph" "fr_fr" "ga_ie" "gl_es" "gu_in" "ha_ng" "he_il" "hi_in" "hr_hr" "hu_hu" "hy_am" "id_id" "ig_ng" "is_is" "it_it" "ja_jp" "jv_id" "ka_ge" "kam_ke" "kea_cv" "kk_kz" "km_kh" "kn_in" "ko_kr" "ky_kg" "lb_lu" "lg_ug" "ln_cd" "lo_la" "lt_lt" "luo_ke" "lv_lv" "mi_nz" "mk_mk" "ml_in" "mn_mn" "mr_in" "ms_my" "mt_mt" "my_mm" "nb_no" "ne_np" "nl_nl" "nso_za" "ny_mw" "oc_fr" "om_et" "or_in" "pa_in" "pl_pl" "ps_af" "pt_br" "ro_ro" "ru_ru" "sd_in" "sk_sk" "sl_si" "sn_zw" "so_so" "sr_rs" "sv_se" "sw_ke" "ta_in" "te_in" "tg_tj" "th_th" "tr_tr" "uk_ua" "umb_ao" "ur_pk" "uz_uz" "vi_vn" "wo_sn" "xh_za" "yo_ng" "yue_hant_hk" "zu_za" 
# for lang in "ast_es" "kam_ke" "kea_cv" "lb_lu" "ms_my" "oc_fr" "ne_np" "lo_la" "umb_ao" "xh_za" "so_so"
# for lang in "kea_cv"
for lang in "ne_np"
do
    sbatch --output="${project_root}/logs/${MODEL}/${timestamp}/%j-%x.out" \
           --error="${project_root}/logs/${MODEL}/${timestamp}/%j-%x.err" \
           -J "${lang}-${MODEL}-transcribe" \
           -t 04:00:00 --mem=32GB --gres="gpu:l40s:1" \
           "${project_root}/scripts/transcribe.sh" "${MODEL}" "${lang}" 1
    sleep 0.1
    sbatch --output="${project_root}/logs/${MODEL}/%j-%x.out" \
           --error="${project_root}/logs/${MODEL}/%j-%x.err" \
           -J "${lang}-${MODEL}-translate" \
           -t 04:00:00 --mem=32GB --gres="gpu:l40s:1" \
           "${project_root}/scripts/transcribe.sh" "${MODEL}" "${lang}" 1 translate
    sleep 0.1
done
