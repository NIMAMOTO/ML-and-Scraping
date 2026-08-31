import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from google import genai
from pydantic import BaseModel, Field
import pandas as pd

#url_2は神戸新聞の一覧ページのJSONリンク
url_ichiran_json = "https://www.kobe-np.co.jp/news/jiken/contentslink/contentslink_list.json?request=/news/jiken/contentslink"

#urlからデータを取ってくる

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
response_ichiran_json = requests.get(url_ichiran_json, headers=headers, timeout=30)
json_data = json.loads(response_ichiran_json.text) #JSONの構造をjson.loadsで戻す

list_ = []
#10件を確実に取る：無料フィルタpay=0→title,time,url→urljoin→request→
for x in json_data["list"]:
    if x["pay"] == "0":
        url_full = urljoin("https://www.kobe-np.co.jp/news/jiken/", x["url"])
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        page_syosai = requests.get(url_full, headers=headers, timeout=30) #作ったfullurlで詳細ページに入ってデータを取ってくる
        time.sleep(0.5)
        soup_syosai = BeautifulSoup(page_syosai.text, "html.parser") #取ってきたデータを解析する
        honbun =soup_syosai.find("div", class_="article-body") #divを掴んで、article-bodyまで範囲を絞る
        honbun_p = honbun.select("p")

        honbun_text = []
        for p in honbun_p:
            p = p.get_text(strip=True)
            honbun_text.append(p)
        honbun_text = ''.join(honbun_text)

        list_.append({"title":x["title"], "time":x["dateDelivery_all"], "url_rel":x["url"], "本文":honbun_text})

    if len(list_) >= 10:
        break


class ArticleScore(BaseModel):
    stage: str = Field(description="第一報／処理／続報 のいずれか")

    a_reason: str = Field(description="Aの根拠。本文からの引用を含む")
    a_1: int = Field(description="人的被害。8/20/40/55/75")
    a_2: int = Field(description="波及規模。30–45／25–40／10–20")
    a_3: int = Field(description="物的被害の加点。0-25")
    a_score: int = Field(description="max(a_1, a_2) + a_3 （上限 100）")
    
    b_reason: str = Field(description="Bの根拠。本文からの引用を含む")
    b_rule: str   = Field(description="該当したB軸のルール項目名を、原文のまま記載")
    b_range: str  = Field(description="そのルールの点数帯。例: 70–90")
    b_score: int = Field(description="帰責性・社会的非難度 0-100")

    c_reason: str = Field(description="Cの根拠。本文からの引用を含む")
    c_1: int = Field(description="被害者属性の加点。0/10/25")
    c_2: int = Field(description="場所の加点。0/15")
    c_3: int = Field(description="全国的文脈の加点。0/20")
    c_4: int = Field(description="地域課題の加点。0/15")
    c_score: int = Field(description="25 + c_1 + c_2 + c_3 + c_4")

    d_reason: str = Field(description="Dの根拠。本文からの引用を含む")
    d_rule: str   = Field(description="該当したD軸のルール項目名を、原文のまま記載")
    d_range: str  = Field(description="そのルールの点数帯。例: 65-85")
    d_score: int  = Field(description="d_range の範囲内の整数")

    summary: str = Field(description="記事の要約。2〜3文。いつ・どこで・誰が・何が起きたか、及び現時点での処理状況（逮捕、調査中、死亡確認等）を含めること。採点の説明ではなく、記事そのものの要約を書くこと。")

ScoringRules = """
あなたは地方紙（神戸新聞「事件・事故」欄）の分析者です。
以下の記事本文に基づき、当該事件の「社会的影響力」を評価し、0〜100点で出力してください。

【前提】
・评估对象：兵庫県内で発生・処理された事件・事故（犯罪、火災、
  交通事故、労働災害、自然災害、サイバー犯罪を含む）
・"社会影响力" = 该事件在多大程度上会引发続報、県内での関心持続、
  再発防止の議論、制度・運用の見直し。不等于被害规模本身。
・多数の記事は 10–45 点に収まる。50 点超は稀。安易に高得点を
  付けないこと。
・本文は 100–300 字の短報である。情報の欠落は文体上の通例であり、
  減点理由としない。明示的な記述がある場合のみ加点する。
・本文の記述のみに基づくこと。自身の既有知識で補完しない。

【記事段階の判定（最初に行う）】
・第一報型：発生直後、原因・被疑者が未確定
・処理型　：逮捕・鎮火など一次的な決着がついたもの
・続報型　：既報事案のその後（送検、鑑定留置、公判、行政処分等）
  → 続報型は「事件全体がまだ未完結」であることを意味するため、
    D を高めに評価する

【四つの分項】      
【A. 被害・影響規模（0–100）】
  計算式： A = max(A-1, A-2) + A-3 （上限 100）
  A-1 人的被害
    死亡 …… 75／意識不明の重体 …… 55／重傷 …… 40
    軽傷 …… 20／人的被害なし …… 8
    複数被害者は最重症者を基準に、人数に応じ +5〜15
  A-2 波及規模
    数万人規模の生活影響（鉄道の運転見合わせ、大規模停電、
      断水、道路の長時間通行止め） …… 45–60
    数千人規模・地域限定の影響 …… 30–45
    複数世帯・複数棟への延焼、複数車両 …… 25–40
    単一世帯・単一車両のみ …… 10–20
    ※金銭被害のみの事案（詐欺等）はここで評価する
      1千万円以上 …… 40–55／百万〜1千万円 …… 30–40
      百万円未満 …… 15–30
  A-3 物的被害（加算）
    建物全焼1棟 +10／延焼で複数棟 +15
    その他、規模に応じ +0〜10

【B. 帰責性・社会的非難度（0–100）】
  B-1 帰責構造
    ・偶発・不可抗力（動物との接触、自然現象） …… 10–25
    ・被害者本人の過失で本人のみが被害 …… 5–20
    ・本人の過失で第三者に被害 …… 40–60
    ・事業者・管理者の安全管理義務違反が示唆される …… 70–90
      （安全帯・保護具の未装着、単独作業、点検不備、
       基準違反、以前から指摘されていた等）
    ・隠蔽・虚偽報告・行政の不作為 …… 90–100
  B-2 犯罪類型の非難度
  ・殺人・強盗致死 …… 90–100
  ・性犯罪（被害者が未成年の場合は上限側） …… 75–90
  ・児童虐待・監護者による犯罪 …… 80–95
  ・放火／不審火で放火の疑いが示唆される …… 60–80
  ・特殊詐欺・組織的詐欺 …… 55–80
     ※高齢者が被害者の場合は上限側。被害額により調整
  ・ストーカー・DV・つきまとい …… 60–75
  ・飲酒運転・無免許運転・あおり運転（人身被害あり） …… 60–75
  ・飲酒運転・無免許運転・あおり運転（人身被害なし） …… 45–60
  ・薬物事犯 …… 45–60
  ・公務員・教員・警察官等による犯罪 …… 60–80
     ※職務上の信頼を裏切る性質のため、罪種より高く評価
  ・不正アクセス等（私的動機によるもの） …… 45–60
  ・一般的な窃盗・万引き …… 25–40

【C. 地域的共鳴・話題性（0–100）】
  基準点 25 から出発し、以下の該当項目を加算する（上限 100）。
  該当がなければ 25 のままとする。県内の日常的な事件・事故の
  大半は 25–45 に収まる。

  C-1 被害者属性 ※Aで測る被害の程度とは別に、属性による
      共鳴のみを評価する
    ・子ども・生徒・園児が被害者 …… +25
    ・独居高齢者・要介護者が被害者 …… +10
    ・被害者が特定されていない／成人一般 …… +0
  C-2 場所
    ・通学路・学校・住宅街・公共交通・商業施設など、
      読者が日常的に利用する場所 …… +15
    ・甲子園球場、有名観光地、著名施設など象徴性のある場所…… +15
    ・私有地・山林・事業所内など、限定的な場所 …… +0
    ※両方に該当する場合は高い方のみ
  C-3 全国的文脈との接続
    ・全国大会、万博、著名キャラクター、全国的に議論されて
      いる問題（転売、あおり運転、特殊詐欺、カスハラ等）と
      接続する …… +20
    ・接続しない …… +0
  C-4 県内共通の地域課題との接続
    以下のいずれかに明示的に該当する場合のみ +15。
    拡大解釈しないこと。
      獣害（シカ・イノシシ・クマ）／過疎地の交通・医療／
      独居高齢者の生活事故／空き家・老朽建築物／
      高齢者の農作業・野焼き／観光地の安全管理／
      外国人住民に関わる事案
    ・上記に該当しない …… +0

【D. 未完結性・継続リスク（0–100）】
  この軸が測るのは「事案が社会的に継続していること」である。
  以下は第一報の定型表現であり、それ自体は加点根拠としない：
    「〜とみて調べている」「出火原因を調べている」
    「〜が原因とみられる」
  ・重大事案の被疑者が逃走中・未特定 …… 70–90
     ※対象は B が 60 以上の事案に限る
      （殺人、傷害、性犯罪、放火、ひき逃げ等）
  ・司法プロセス進行中（鑑定留置、公判、責任能力・
    認否が争点、控訴） …… 65–85
  ・意識不明の重体（容体が変化しうる） …… 60–75
  ・行政調査・処分が想定される
    （労基署、監督官庁、営業停止、免許取消） …… 60–75
  ・同種事案の連続・多発が示唆される …… 60–80
  ・軽微事案で原因調査中（定型句のみ） …… 25–35
  ・逮捕済み・容疑を認めており争点なし …… 15–30
  ・鎮火済み・軽傷・原因判明 …… 10–20
  【上限規則】
   A < 20 かつ B < 40 の場合、D は 40 を上限とする。
   被害も非難度も低い事案は、調査継続中であっても
   社会影響力は低い。
"""

csv =[]
for x in list_:
  fullprompt = ScoringRules + "\n\n【配信日時】\n" + x["time"] + "\n\n【記事本文】\n" + x["本文"]

  client = genai.Client()
  interaction = client.interactions.create(
      model = "gemini-3.6-flash",
      input = fullprompt,
      response_format={
        "type":"text",
        "mime_type":"application/json",
        "schema":ArticleScore.model_json_schema(),
        "temperature": 0.1,
        "timeout": 10000
      })
  articlescore = ArticleScore.model_validate_json(interaction.output_text)
  print("*** Article Score ***",articlescore)

  A = min(max(articlescore.a_1, articlescore.a_2) + articlescore.a_3, 100)
  C = min(25 + articlescore.c_1 + articlescore.c_2 + articlescore.c_3 + articlescore.c_4, 100)

  TotalScore = (
     (A * 0.30) +
     (articlescore.b_score * 0.30) +
      (C * 0.20) +
      (min(articlescore.d_score * 0.20, 40) if (A * 0.30 < 20 and articlescore.b_score * 0.30 < 40) else (articlescore.d_score * 0.20))
  )

  csv.append({
      "タイトル": x["title"],
      "リスクスコア": TotalScore,
      "要約": articlescore.summary,
      "本文": x["本文"]
      })
  
  print(f"[{len(csv)}] Total={TotalScore:.1f}  A={A}(AI:{articlescore.a_score})  B={articlescore.b_score}{articlescore.b_range}  C={C}(AI:{articlescore.c_score})  D={articlescore.d_score}{articlescore.d_range}")
  

CSV = pd.DataFrame(csv, columns=["タイトル","リスクスコア","要約","本文"])
CSV.to_csv("kobe_news_riskscore.csv", index=False, encoding="utf-8-sig")


