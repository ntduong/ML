'''
Created on Jan 14, 2012
@author: ntd
'''

import urllib2
from BeautifulSoup import *
from urlparse import urljoin
from sqlite3 import dbapi2 as sqlite

ignorewords=set(['the', 'of', 'to', 'and', 'a', 'in', 'is', 'it'])

class crawler:
    def __init__(self, dbname):
        self.conn = sqlite.connect(dbname)
    
    def __del__(self):
        self.conn.close()
        
    def dbcommit(self):
        self.conn.commit()
        
    # get entry-id: the first column content
    def getentryid(self, table, field, value, createnew=True):
        cur = self.conn.execute("select rowid from %s where %s='%s'" %(table, field, value))
        res = cur.fetchone()
        if res == None:
            cur = self.conn.execute("insert into %s (%s) values ('%s')" %(table, field,value))
            return cur.lastrowid
        else:
            return res[0]
        
    # index page(url)
    def addtoindex(self, url, soup):
        if self.isindexed(url): return
        print 'Indexing %s...' % url
        
        text=self.gettextonly(soup)
        words=self.seperatewords(text)
        
        urlid=self.getentryid('urllist', 'url', url)
        for i in range(len(words)):
            word = words[i]
            if word in ignorewords: continue
            wordid=self.getentryid('wordlist', 'word', word)
            self.conn.execute("insert into wordlocation(urlid, wordid, location)\
            values (%d,%d,%d)" %(urlid, wordid, i))
    
    # get text (no tag) only from HTML content
    def gettextonly(self, soup):
        v = soup.string
        if v == None:
            c = soup.contents
            resulttext=''
            for t in c:
                subtext = self.gettextonly(t)
                resulttext += subtext + '\n'
            return resulttext
        else:
            return v.strip()
        
    
    # split words
    def seperatewords(self, text):
        splitter = re.compile('\\W*')
        return [s.lower() for s in splitter.split(text) if s != '']
        
    # check if a page(url) is indexed or not
    def isindexed(self, url):
        u = self.conn.execute("select rowid from urllist where url='%s'" %url).fetchone()
        if u != None:
            v = self.conn.execute('select * from wordlocation where urlid=%d' %u[0]).fetchone()
            if v != None: return True
        return False
    
    # add link between 2 pages(urls)
    def addlinkref(self, urlFrom, urlTo, linkText):
        words = self.seperatewords(linkText)
        fromid = self.getentryid('urllist', 'url', urlFrom)
        toid = self.getentryid('urllist', 'url', urlTo)
        if fromid == toid: return
        cur = self.conn.execute("insert into link(fromid, toid) values(%d, %d)" %(fromid, toid))
        linkid = cur.lastrowid
        for word in words:
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist', 'word', word)
            self.conn.execute("insert into linkwords(linkid, wordid) values(%d, %d)" %(linkid,wordid))
    
    # crawling by BFS(depth = 2 by default) from list of page(pages)
    def crawl(self, pages, depth=2):
        for i in range(depth):
            newpages=set()
            for page in pages:
                try:
                    c=urllib2.urlopen(page)
                except:
                    print 'Could not open %s' %page
                    continue
                soup = BeautifulSoup(c.read())
                self.addtoindex(page, soup)
                
                links = soup('a')
                for link in links:
                    if 'href' in dict(link.attrs):
                        url = urljoin(page, link['href'])
                        if url.find("'") != -1: continue
                        url = url.split('#')[0]
                        if url[0:4] == 'http' and not self.isindexed(url):
                            newpages.add(url)
                        linkText = self.gettextonly(link)
                        self.addlinkref(page, url, linkText)
                self.dbcommit()
            pages = newpages
    
    def createindextables(self):
        self.conn.execute('create table urllist(url)')
        self.conn.execute('create table wordlist(word)')
        self.conn.execute('create table wordlocation(urlid, wordid, location)')
        self.conn.execute('create table link(fromid integer, toid integer)')
        self.conn.execute('create table linkwords(wordid, linkid)')
        
        # create indexes for speeding up searching operation
        self.conn.execute('create index wordidx on wordlist(word)')
        self.conn.execute('create index urlidx on urllist(url)')
        self.conn.execute('create index wordurlidx on wordlocation(wordid)')
        self.conn.execute('create index urltoidx on link(toid)')
        self.conn.execute('create index urlfromidx on link(fromid)')
        self.dbcommit()
    
class searcher:
    def __init__(self, dbname):
        self.conn = sqlite.connect(dbname)
    
    def __del__(self):
        self.conn.close()
        
    def dbcommit(self):
        self.conn.commit()
        
    def getmatchrow(self, q):
        fieldlist = 'w0.urlid'
        tablelist = ''
        clauselist = ''
        wordids = []
        
        words = q.split(' ')
        tablenumber = 0
        for word in words:
            wordrow = self.conn.execute(
                        "select rowid from wordlist where word = '%s'" %(word)).fetchone()
            if wordrow != None:
                wordid = wordrow[0]
                wordids.append(wordid)
                if tablenumber > 0:
                    tablelist += ','
                    clauselist += ' and '
                    clauselist += 'w%d.urlid=w%d.urlid and ' %(tablenumber-1, tablenumber)
                fieldlist += ',w%d.location' % tablenumber
                tablelist += 'wordlocation w%d' %tablenumber
                clauselist += 'w%d.wordid=%d' %(tablenumber, wordid)
        
        fullquery = 'select %s from %s where %s' %(fieldlist,tablelist,clauselist)
        cur = self.conn.execute(fullquery)
        rows = [row for row in cur]
        return rows, wordids
            
    def getscoredlist(self, rows, wordids):
        totalscores = dict([row[0],0] for row in rows)
        
        weights = [(1.0,self.frequencyscore(rows)),
                   (1.0, self.locationscore(rows)),
                   (1.0, self.pagerankscore(rows))]
        
        for (weight, scores) in weights:
            for url in totalscores:
                totalscores[url] += weight*scores[url]
        return totalscores
    
    def geturlname(self, id):
        return self.conn.execute("select url from urllist where rowid=%d" % id).fetchone()[0]
    
    def query(self, q):
        rows, wordids = self.getmatchrow(q)
        scores = self.getscoredlist(rows, wordids)
        rankedscores = sorted([(score,url) for (url, score) in scores.items()], reverse=True)
        for (score, urlid) in rankedscores[0:10]:
            print '%f\t%s' % (score, self.geturlname(urlid))
    
    # make score value to be in [0,1]
    # below scores = {(url, score)}
    def normalizescores(self, scores, smallIsBetter=False):
        vsmall = 0.00001 # avoid divide by 0
        if smallIsBetter:
            minscore = min(scores.values())
            return dict([(u, float(minscore)/max(vsmall,l)) for (u,l) in scores.items()])
        else:
            maxscore = max(scores.values())
            if maxscore == 0: maxscore = vsmall
            return dict([(u, float(c)/maxscore) for (u,c) in scores.items()])
        
    def frequencyscore(self, rows):
        counts = dict([(row[0], 0) for row in rows])
        for row in rows: counts[row[0]] += 1
        return self.normalizescores(counts)
    
    
    def locationscore(self, rows):
        locations = dict([(row[0], 1000000) for row in rows])
        for row in rows:
            loc = sum(row[1:])
            if loc < locations[row[0]]: locations[row[0]] = loc
        return self.normalizescores(locations, smallIsBetter=True)
    
    def distancescore(self, rows):
        if len(rows[0]) <= 2: return dict([(row[0],1.0) for row in rows])
        
        mindistance = dict([(row[0], 1000000) for row in rows])
        
        for row in rows:
            dist = sum([abs(row[i]-row[i-1]) for i in range(2, len(row))])
            if dist < mindistance[row[0]]: mindistance[row[0]] = dist
        return self.normalizescores(dist, smallIsBetter=True)

    def inboundlinkscore(self, rows):
        uniqueurls = set([row[0] for row in rows])
        inboundcount = dict([(u,self.conn.execute('select count(*) from link where toid=%d' % u).fetchone()[0])
                             for u in uniqueurls])
        
        return self.normalizescores(inboundcount, smallIsBetter=False)

    def calculatepagerank(self, iterations=20):
        self.conn.execute('drop table if exists pagerank')
        self.conn.execute('create table pagerank(urlid primary key, score)')
        
        self.conn.execute('insert into pagerank select rowid, 1.0 from urllist')
        self.dbcommit()
        
        for i in range(iterations):
            print 'Iteration %d' % i
            for(urlid,) in self.conn.execute('select rowid from urllist'):
                pr = 0.15
                
                for (linker,) in self.conn.execute('select distinct fromid from link where toid=%d' %urlid):
                    linkingpr = self.conn.execute('select score from pagerank where urlid=%d' %linker).fetchone()[0]
                    linkingcount = self.conn.execute('select count(*) from link where fromid=%d' %linker).fetchone()[0]
                    pr += 0.85*(linkingpr/linkingcount)
                self.conn.execute('update pagerank set score=%f where urlid=%d' %(pr,urlid))
                self.dbcommit()
    
    def pagerankscore(self, rows):
        pageranks = dict([(row[0], self.conn.execute('select score from pagerank where urlid=%d' %row[0]).fetchone()[0])
                          for row in rows])
        maxrank = max(pageranks.values())
        normalizedscores = dict([(u, float(l)/maxrank) for (u,l) in pageranks.items()])
        
        return normalizedscores
    
    def linktextscore(self, rows, wordids):
        linkscores = dict([(row[0], 0) for row in rows])
        for wordid in wordids:
            cur = self.conn.execute('select link.fromid, link.toid from linkwords, link where \
            wordid=%d and linkwords.linkid=link.rowid' %wordid)
            for (fromid, toid) in cur:
                if toid in linkscores:
                    pr = self.conn.execute('select score from pagerank where urlid=%d' %fromid).fetchone()[0]
                linkscores[toid] += pr
        maxscore = max(linkscores.values())
        normalizedscores = dict([(u, float(l)/maxscore) for (u,l) in linkscores.items()])
        return normalizedscores

if __name__ == '__main__':
    '''
    test_crawler = crawler('searchindex.db')
    test_crawler.createindextables()
    pages = ['http://www.titech.ac.jp']
    test_crawler.crawl(pages)
    '''
    test_searcher = searcher('searchindex.db')
    test_searcher.calculatepagerank()
    test_searcher.query('schedule 2012')