import json
import argparse

def analysis_main(preds, xml_scores):
    rr = 0
    rw = 0
    wr = 0
    ww = 0
    for item1, item2 in zip(preds, xml_scores):
        if item1 == 0 and item2 == 0:
            ww += 1
        elif item1 == 0 and item2 == 1:
            wr += 1
        elif item1 == 1 and item2 == 0:
            rw += 1
        elif item1 == 1 and item2 == 1:
            rr += 1
    return rr / len(preds), rw / len(preds), wr / len(preds), ww / len(preds)

def main(args):
    with open(args.anno_json_path, 'r') as file:
        anno = json.load(file)
    with open(args.zs_pred_json_path, 'r') as file:
        pred_zs = json.load(file)
        pred_zs_01 = [1 if x == y else 0 for x, y in zip(pred_zs, anno)]
    with open(args.zs_rationale_json_path, 'r') as file:
        rma_zs = json.load(file)
    with open(args.lp_pred_json_path, 'r') as file:
        pred_lp = json.load(file)
        pred_lp_01 = [1 if x == y else 0 for x, y in zip(pred_lp, anno)]
    with open(args.lp_rationale_json_path, 'r') as file:
        rma_lp = json.load(file)
    with open(args.flcp_pred_json_path, 'r') as file:
        pred_flcp = json.load(file)
        pred_flcp_01 = [1 if x == y else 0 for x, y in zip(pred_flcp, anno)]
    with open(args.flcp_rationale_json_path, 'r') as file:
        rma_flcp = json.load(file)
    with open(args.ft_pred_json_path, 'r') as file:
        pred_ft = json.load(file)
        pred_ft_01 = [1 if x == y else 0 for x, y in zip(pred_ft, anno)]
    with open(args.ft_rationale_json_path, 'r') as file:
        rma_ft = json.load(file)

    res_zs = analysis_main(preds=pred_zs_01, xml_scores=rma_zs)
    res_lp = analysis_main(preds=pred_lp_01, xml_scores=rma_lp)
    res_flcp = analysis_main(preds=pred_flcp_01, xml_scores=rma_flcp)
    res_ft = analysis_main(preds=pred_ft_01, xml_scores=rma_ft)

    res_zs_final = {}
    res_lp_final = {}
    res_flcp_final = {}
    res_ft_final = {}

    res_zs_final['PT'] = round(res_zs[0] / (res_zs[0] + res_zs[1]), 5)
    res_lp_final['PT'] = round(res_lp[0] / (res_lp[0] + res_lp[1]), 5)
    res_flcp_final['PT'] = round(res_flcp[0] / (res_flcp[0] + res_flcp[1]), 5)
    res_ft_final['PT'] = round(res_ft[0] / (res_ft[0] + res_ft[1]), 5)

    res_zs_final['IR'] = round(res_zs[0] / (res_zs[0] + res_zs[2]), 5)
    res_lp_final['IR'] = round(res_lp[0] / (res_lp[0] + res_lp[2]), 5)
    res_flcp_final['IR'] = round(res_flcp[0] / (res_flcp[0] + res_flcp[2]), 5)
    res_ft_final['IR'] = round(res_ft[0] / (res_ft[0] + res_ft[2]), 5)

    print("ZS Results:\n", res_zs_final, '\n')
    print("LP Results:\n", res_lp_final, '\n')
    print("FLCP Results:\n", res_flcp_final, '\n')
    print("FT Results:\n", res_ft_final, '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--anno_json_path', type=str,
                        help="Path of prediction annotations for one dataset.")
    parser.add_argument('--zs_pred_json_path', type=str,
                        help="Path of prediction results for ZS method on one dataset.")
    parser.add_argument('--zs_rationale_json_path', type=str,
                        help="Path of rationale results for ZS method on one dataset.")
    parser.add_argument('--lp_pred_json_path', type=str,
                        help="Path of prediction results for LP method on one dataset.")
    parser.add_argument('--lp_rationale_json_path', type=str,
                        help="Path of rationale results for LP method on one dataset.")
    parser.add_argument('--flcp_pred_json_path', type=str,
                        help="Path of prediction results for FLCP method on one dataset.")
    parser.add_argument('--flcp_rationale_json_path', type=str,
                        help="Path of rationale results for FLCP method on one dataset.")
    parser.add_argument('--ft_pred_json_path', type=str,
                        help="Path of prediction results for FT method on one dataset.")
    parser.add_argument('--ft_rationale_json_path', type=str,
                        help="Path of rationale results for FT method on one dataset.")
    args = parser.parse_args()

    main(args)
