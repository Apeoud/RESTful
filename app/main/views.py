import json
import traceback

from . import main
from flask import request, jsonify

from app.RelationExtraction.dataStructure.relation import AUTORE


@main.route('/test/<username>')
def hello(username):
    return 'service test successful, ' + username


@main.route('/relation', methods=['POST'])
def relation_extraction():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, 'utf-8')
        task = json_request_data["task"]
        if task == "re":
            re = AUTORE(config_file='/Users/duanshangfu/PycharmProjects/RESTful/app/RelationExtraction/parameter.json')
            # re.bootstrap(sentence_file='sentences.txt')
            re.load_model()
            sentence = json_request_data["sentence"]
            relation = re.score(sentence)
            return jsonify({
                'result': [str(key) + str(value) for key, value in relation.items()]
            })
    except Exception as e:
        return jsonify({
            "ret_code": -1,
            "err_info": traceback.format_exc()
        })
