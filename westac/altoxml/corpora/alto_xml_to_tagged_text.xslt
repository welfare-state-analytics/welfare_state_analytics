<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="delimiter"/>
<xsl:param name="target"/>

<xsl:template match="w">

    <xsl:variable name="lemma" select="@lemma"/>
    <xsl:variable name="content_token" select="text()"/>
    <xsl:variable name="lemma_token" select="substring-before(substring-after($lemma,'|'),'|')"/>

    <xsl:choose>
        <xsl:when test="$target='lemma' and $lemma_token!=''"><xsl:value-of select="$lemma_token"/></xsl:when>
        <xsl:otherwise><xsl:value-of select="$content_token"/><xsl:value-of select="$content_token"/></xsl:otherwise>
    </xsl:choose>

    <xsl:text>/</xsl:text><xsl:value-of select="@pos"/><xsl:text> </xsl:text>

</xsl:template>

</xsl:stylesheet>