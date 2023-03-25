class _ParsedToken:
    def __init__(self, identifier):
        self.identifier = identifier
        self.children = []
        
    def hasChildren(self):
        return len(self.children) > 0
    
    def addChild(self, child):
        self.children.append(child)
        
    def removeChildrenFrom(self, parentId, childId):
        if self.identifier == parentId:
            self.children = [child for child in self.children if child.identifier != childId]
        
        for child in self.children:
            child.removeChildrenFrom(parentId, childId)

    def wrapChildrenOf(self, parentId, wrapId):
        if self.identifier == parentId:
            existingChildren = self.children
            self.children = []
            for child in existingChildren:
                token = _ParsedToken(wrapId)
                token.addChild(child)
                self.addChild(token)

        for child in self.children:
            child.wrapChildrenOf(parentId, wrapId)
         
    def removeIdentifierPromoteChildren(self, id):
        prevChildren = self.children
        self.children = []

        for child in prevChildren:
            if child.identifier == id:
                for grandchild in child.children:
                    self.addChild(grandchild)
            else:
                self.addChild(child)

        for child in self.children:
            child.removeIdentifierPromoteChildren(id)

    def renameIdentifier(self, src, dest):
        if self.identifier == src:
            self.identifier = dest
            
        for child in self.children:
            child.renameIdentifier(src, dest)

    def findIdentifiers(self, id, matches):
        if self.identifier == id:
            matches.append(self)

        for child in self.children:
            matches = child.findIdentifiers(id, matches)

        return matches
        
    def reconstruct(self):
        result = self.identifier
        
        if len(self.children) > 0:
            result += "("
        
        for i in range(0, len(self.children)):
            result += self.children[i].reconstruct()
            if i != len(self.children) - 1:
                result += ", "
            
        if len(self.children) > 0:
            result += ")"
        
        return result

    def __repr__(self):
        if len(self.children) > 0:
            return f"{self.identifier}:{self.children}"
        else:
            return f"{self.identifier}"

        
    def __str__(self):
        return self.__repr__()

def _scanForOpeningParenthesis(input):
    idx = 0
    
    while idx < len(input):
        if input[idx] == "(":
            return idx
        idx = idx + 1
    
    return -1

def _scanForClosingParenthesis(input):
    stack = []
    
    idx = 0
    while idx < len(input):
        if input[idx] == "(":
            stack.append("(")
        elif input[idx] == ")" and stack[-1] == "(":
            stack.pop()
        if len(stack) == 0:
            return idx
        idx = idx + 1
    
    return -1

def _splitOnCommas(input):
    result = []
    lastStart = 0
    stack = []
    
    idx = 0
    while idx < len(input):
        if input[idx] == "(":
            stack.append("(")
        elif input[idx] == ")" and stack[-1] == "(":
            stack.pop()
        if len(stack) == 0 and input[idx] == ",":
            result.append(input[lastStart:idx])
            lastStart = idx + 1
        idx = idx + 1
    
    result.append(input[lastStart:len(input)])
    
    return result
    
def _parse(expression):
    openingParenthesisIdx = _scanForOpeningParenthesis(expression)
    closingParenthesisIdx = _scanForClosingParenthesis(expression[openingParenthesisIdx:]) + openingParenthesisIdx
    
    if (openingParenthesisIdx == -1 and closingParenthesisIdx == -1) \
            or (expression.startswith("'") and expression.endswith("'") and expression.count("'") == 2 \
            and expression.count('"') == 0) \
            or (expression.startswith('"') and expression.endswith('"') and expression.count('"') == 2 \
            and expression.count("'") == 0):
        return _ParsedToken(expression)
    
    identifier = expression[0:openingParenthesisIdx]
    
    params = _splitOnCommas(expression[openingParenthesisIdx + 1:closingParenthesisIdx])

    token = _ParsedToken(identifier)
    
    for param in params:
        token.addChild(_parse(param.strip()))

    assert token.reconstruct() == expression
    
    return token

